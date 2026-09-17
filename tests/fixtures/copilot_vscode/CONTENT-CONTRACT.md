# Sanitized core chat-v3 transcript subset

Task 4 recognizes only these invented, pinned fixture shapes: request text at
`requests[n].message.text`; response parts `markdownContent.content.value`,
`thinking.value`, `toolInvocation` with string `toolCallId`, `toolName`,
`arguments`, and optional string `result`; and `attachment` with string `name`
and optional URL. Unsupported parts become fixed notices. URLs are labels only.
