package bedrockclient

import (
	"bytes"
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/tmc/langchaingo/llms"
)

// Client is a Bedrock client.
type Client struct {
	client *bedrockruntime.Client
}

// NewClient creates a new Bedrock client.
func NewClient(client *bedrockruntime.Client) *Client {
	return &Client{
		client: client,
	}
}

// CreateCompletion creates a new completion response from the provider
// after sending the messages to the provider.
func (c *Client) CreateCompletion(ctx context.Context,
	modelID string,
	messages []llms.MessageContent,
	options llms.CallOptions,
) (*llms.ContentResponse, error) {
	inferenceConfig := &types.InferenceConfiguration{
		MaxTokens:     aws.Int32(int32(getMaxTokens(options.MaxTokens, 512))),
		TopP:          aws.Float32(float32(options.TopP)),
		Temperature:   aws.Float32(float32(options.Temperature)),
		StopSequences: options.StopWords,
	}

	systemMessages, otherMessages := []llms.MessageContent{}, []llms.MessageContent{}
	for _, m := range messages {
		if m.Role == llms.ChatMessageTypeSystem {
			systemMessages = append(systemMessages, m)
		} else {
			otherMessages = append(otherMessages, m)
		}
	}

	systemPrompt, err := processSystemMessages(systemMessages)
	if err != nil {
		return nil, err
	}

	m, err := processMessages(otherMessages)
	if err != nil {
		return nil, err
	}

	input := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(modelID),
		Messages:        m,
		InferenceConfig: inferenceConfig,
		System:          systemPrompt,
	}

	output, err := c.client.Converse(ctx, input)
	if err != nil {
		return nil, err
	}

	// according to the docs this is always what is returned
	outputMessage, ok := output.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return nil, errors.New("unexpected output type")
	}

	outputContents := []string{}
	for _, content := range outputMessage.Value.Content {
		switch typedContent := content.(type) {
		case *types.ContentBlockMemberText:
			outputContents = append(outputContents, typedContent.Value)
		case *types.ContentBlockMemberImage:
			imageSource := typedContent.Value.Source
			if imageSourceBytes, ok := imageSource.(*types.ImageSourceMemberBytes); ok {
				outputContents = append(outputContents, string(imageSourceBytes.Value))
			}
		}
	}

	return &llms.ContentResponse{
		Choices: []*llms.ContentChoice{
			{
				Content:    strings.Join(outputContents, "\n"),
				StopReason: string(output.StopReason),
				GenerationInfo: map[string]any{
					"input_tokens":  output.Usage.InputTokens,
					"output_tokens": output.Usage.OutputTokens,
				},
			},
		},
	}, nil
}

func processSystemMessages(messages []llms.MessageContent) ([]types.SystemContentBlock, error) {
	if len(messages) == 0 {
		return nil, nil
	}

	if len(messages) > 1 {
		return nil, fmt.Errorf("expected at most one system message, got %d", len(messages))
	}

	systemMessageContent, ok := messages[0].Parts[0].(llms.TextContent)
	if !ok {
		return nil, fmt.Errorf("expected system message to be llms.TextContent, got %T", messages[0].Parts[0])
	}

	return []types.SystemContentBlock{
		&types.SystemContentBlockMemberText{
			Value: systemMessageContent.Text,
		},
	}, nil
}

func processMessages(messages []llms.MessageContent) ([]types.Message, error) {
	outputMessages := make([]types.Message, len(messages))
	for msgIdx, message := range messages {
		role, err := roleToBedrockRole(message.Role)
		if err != nil {
			return nil, err
		}

		content := make([]types.ContentBlock, len(message.Parts))
		for partIdx, part := range message.Parts {
			bedrockContent, err := messageToBedrockContent(part)
			if err != nil {
				return nil, err
			}
			content[partIdx] = bedrockContent
		}

		bedrockMessage := types.Message{
			Role:    role,
			Content: content,
		}
		outputMessages[msgIdx] = bedrockMessage
	}
	return outputMessages, nil
}

// nolint: exhaustive
func roleToBedrockRole(role llms.ChatMessageType) (types.ConversationRole, error) {
	switch role {
	case llms.ChatMessageTypeHuman:
		return types.ConversationRoleUser, nil
	case llms.ChatMessageTypeAI:
		return types.ConversationRoleAssistant, nil
	}
	return "", fmt.Errorf("unsupported role: %s", role)
}

// nolint: exhaustive
func messageToBedrockContent(content llms.ContentPart) (types.ContentBlock, error) {
	switch typedContent := content.(type) {
	case llms.TextContent:
		return &types.ContentBlockMemberText{
			Value: typedContent.Text,
		}, nil
	case llms.BinaryContent:
		return binaryContentToBedrockContent(typedContent)
	case llms.ImageURLContent:
		return imageURLContentToBedrockContent(typedContent)
	}
	return nil, fmt.Errorf("unsupported content type: %T", content)
}

func imageURLContentToBedrockContent(content llms.ImageURLContent) (types.ContentBlock, error) {
	parts := strings.Split(content.URL, ";")
	if len(parts) != 2 {
		return nil, fmt.Errorf("unsupported image url: %s", content.URL)
	}

	mimeType, found := strings.CutPrefix(parts[0], "data:")
	if !found {
		return nil, fmt.Errorf("unsupported image url: %s", content.URL)
	}

	imageFormat, err := mimetypeToBedrockImageFormat(mimeType)
	if err != nil {
		return nil, err
	}

	var b bytes.Buffer
	if strings.HasPrefix(parts[1], "base64,") {
		data, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(parts[1], "base64,"))
		if err != nil {
			return nil, err
		}
		b.Write(data)
	} else {
		b.WriteString(parts[1])
	}

	return &types.ContentBlockMemberImage{
		Value: types.ImageBlock{
			Format: imageFormat,
			Source: &types.ImageSourceMemberBytes{
				Value: b.Bytes(),
			},
		},
	}, nil
}

func binaryContentToBedrockContent(content llms.BinaryContent) (types.ContentBlock, error) {
	imageFormat, err := mimetypeToBedrockImageFormat(content.MIMEType)
	if err == nil {
		return &types.ContentBlockMemberImage{
			Value: types.ImageBlock{
				Format: imageFormat,
				Source: &types.ImageSourceMemberBytes{
					Value: content.Data,
				},
			},
		}, nil
	}

	documentFormat, err := mimetypeToBedrockDocumentFormat(content.MIMEType)
	if err == nil {
		return &types.ContentBlockMemberDocument{
			Value: types.DocumentBlock{
				Name:   content.Filename,
				Format: documentFormat,
				Source: &types.DocumentSourceMemberBytes{
					Value: content.Data,
				},
			},
		}, nil
	}

	return nil, fmt.Errorf("unsupported content type: %T", content)
}

func mimetypeToBedrockImageFormat(mimeType string) (types.ImageFormat, error) {
	switch mimeType {
	case "image/png":
		return types.ImageFormatPng, nil
	case "image/jpeg":
		return types.ImageFormatJpeg, nil
	case "image/gif":
		return types.ImageFormatGif, nil
	case "image/webp":
		return types.ImageFormatWebp, nil
	}
	return "", fmt.Errorf("unsupported mime type: %s", mimeType)
}

func mimetypeToBedrockDocumentFormat(mimeType string) (types.DocumentFormat, error) {
	switch mimeType {
	case "application/pdf":
		return types.DocumentFormatPdf, nil
	case "text/csv":
		return types.DocumentFormatCsv, nil
	case "application/msword":
		return types.DocumentFormatDoc, nil
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return types.DocumentFormatDocx, nil
	case "application/vnd.ms-excel":
		return types.DocumentFormatXls, nil
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return types.DocumentFormatXlsx, nil
	case "text/html":
		return types.DocumentFormatHtml, nil
	case "text/plain":
		return types.DocumentFormatTxt, nil
	case "text/markdown":
		return types.DocumentFormatMd, nil
	}
	return "", fmt.Errorf("unsupported mime type: %s", mimeType)
}

func getMaxTokens(maxTokens, defaultValue int) int {
	if maxTokens <= 0 {
		return defaultValue
	}
	return maxTokens
}
