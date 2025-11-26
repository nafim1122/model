export interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: string;
  updatedAt: string;
}

export interface Settings {
  model: string;
  systemPrompt: string;
  theme: 'dark' | 'light';
}

export interface GenerateRequest {
  prompt: string;
  systemPrompt: string;
  model: string;
  conversationHistory: Message[];
}
