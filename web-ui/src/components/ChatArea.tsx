'use client';

import { useState, useRef, useEffect, KeyboardEvent } from 'react';
import { Send, PanelLeft, Sparkles, Code2, Copy, Check, RefreshCw } from 'lucide-react';
import { Conversation } from '@/types';
import Prism from 'prismjs';
import 'prismjs/components/prism-c';

interface ChatAreaProps {
  conversation: Conversation | undefined;
  isLoading: boolean;
  sidebarOpen: boolean;
  onSendMessage: (content: string) => void;
  onToggleSidebar: () => void;
  onNewChat: () => void;
}

export default function ChatArea({
  conversation,
  isLoading,
  sidebarOpen,
  onSendMessage,
  onToggleSidebar,
  onNewChat,
}: ChatAreaProps) {
  const [input, setInput] = useState('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation?.messages]);

  // Highlight code after render
  useEffect(() => {
    Prism.highlightAll();
  }, [conversation?.messages]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [input]);

  const handleSend = () => {
    if (input.trim() && !isLoading) {
      onSendMessage(input.trim());
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const copyToClipboard = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const extractCodeBlocks = (content: string) => {
    const parts: Array<{ type: 'text' | 'code'; content: string; language?: string }> = [];
    const codeBlockRegex = /```(\w*)\n?([\s\S]*?)```/g;
    let lastIndex = 0;
    let match;

    while ((match = codeBlockRegex.exec(content)) !== null) {
      // Add text before code block
      if (match.index > lastIndex) {
        const textBefore = content.slice(lastIndex, match.index).trim();
        if (textBefore) {
          parts.push({ type: 'text', content: textBefore });
        }
      }
      // Add code block
      parts.push({
        type: 'code',
        content: match[2].trim(),
        language: match[1] || 'c',
      });
      lastIndex = match.index + match[0].length;
    }

    // Add remaining text
    if (lastIndex < content.length) {
      const remainingText = content.slice(lastIndex).trim();
      if (remainingText) {
        parts.push({ type: 'text', content: remainingText });
      }
    }

    // If no code blocks found, check if entire content looks like C code
    if (parts.length === 0 && content.trim()) {
      if (content.includes('#include') || content.includes('int main') || content.includes('void ')) {
        parts.push({ type: 'code', content: content.trim(), language: 'c' });
      } else {
        parts.push({ type: 'text', content: content.trim() });
      }
    }

    return parts;
  };

  const renderMessage = (message: { id: string; role: string; content: string }, index: number) => {
    const isUser = message.role === 'user';
    const parts = isUser ? [{ type: 'text' as const, content: message.content }] : extractCodeBlocks(message.content);

    return (
      <div
        key={message.id}
        className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-fade-in`}
      >
        <div
          className={`max-w-[85%] ${
            isUser
              ? 'bg-user-bubble text-white rounded-2xl rounded-tr-sm px-4 py-3'
              : 'bg-transparent'
          }`}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="space-y-3">
              {parts.map((part, partIndex) => (
                <div key={partIndex}>
                  {part.type === 'text' ? (
                    <p className="text-gray-200 dark:text-gray-200 whitespace-pre-wrap leading-relaxed">
                      {part.content}
                    </p>
                  ) : (
                    <div className="rounded-lg overflow-hidden bg-[#1e1e1e] border border-gray-700">
                      <div className="flex items-center justify-between px-4 py-2 bg-[#2d2d2d] border-b border-gray-700">
                        <div className="flex items-center gap-2">
                          <Code2 className="w-4 h-4 text-accent" />
                          <span className="text-xs text-gray-400 uppercase tracking-wider">
                            {part.language || 'c'}
                          </span>
                        </div>
                        <button
                          onClick={() => copyToClipboard(part.content, `${message.id}-${partIndex}`)}
                          className="flex items-center gap-1 px-2 py-1 text-xs text-gray-400 hover:text-white hover:bg-gray-700 rounded transition-colors"
                          title="Copy code"
                        >
                          {copiedId === `${message.id}-${partIndex}` ? (
                            <>
                              <Check className="w-3 h-3 text-green-500" />
                              <span className="text-green-500">Copied!</span>
                            </>
                          ) : (
                            <>
                              <Copy className="w-3 h-3" />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                      </div>
                      <pre className="p-4 overflow-x-auto">
                        <code className={`language-${part.language || 'c'}`}>
                          {part.content}
                        </code>
                      </pre>
                    </div>
                  )}
                </div>
              ))}
              {isLoading && index === (conversation?.messages.length ?? 0) - 1 && message.role === 'assistant' && (
                <span className="typing-cursor"></span>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-chat-dark dark:bg-chat-dark">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800 bg-chat-dark">
        <div className="flex items-center gap-3">
          {!sidebarOpen && (
            <button
              onClick={onToggleSidebar}
              className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
              title="Open sidebar"
            >
              <PanelLeft className="w-5 h-5" />
            </button>
          )}
          <div>
            <h2 className="text-white font-medium">
              {conversation?.title || 'C-Code LLM Studio'}
            </h2>
            <p className="text-xs text-gray-500">
              {conversation ? `${conversation.messages.length} messages` : 'AI-powered C code generation'}
            </p>
          </div>
        </div>
        {!sidebarOpen && (
          <button
            onClick={onNewChat}
            className="flex items-center gap-2 px-3 py-2 bg-accent hover:bg-accent-hover text-white text-sm rounded-lg transition-colors"
          >
            <Sparkles className="w-4 h-4" />
            New Chat
          </button>
        )}
      </div>

      {/* Messages Area */}
      <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-4">
        {!conversation || conversation.messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center px-4">
            <div className="w-20 h-20 bg-gradient-to-br from-accent to-blue-600 rounded-2xl flex items-center justify-center mb-6 shadow-lg shadow-accent/20">
              <Code2 className="w-10 h-10 text-white" />
            </div>
            <h2 className="text-2xl font-bold text-white mb-2">C-Code LLM Studio</h2>
            <p className="text-gray-400 mb-8 max-w-md">
              Your AI assistant for generating clean, efficient C code. 
              Ask me anything about C programming!
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-2xl">
              {[
                'Write a function to reverse a string in C',
                'Create a binary search implementation',
                'Show me how to use pointers with arrays',
                'Implement a linked list in C',
              ].map((suggestion, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setInput(suggestion);
                    textareaRef.current?.focus();
                  }}
                  className="text-left px-4 py-3 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-300 transition-colors border border-gray-700 hover:border-gray-600"
                >
                  <span className="text-accent mr-2">→</span>
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="max-w-4xl mx-auto">
            {conversation.messages.map((message, index) => renderMessage(message, index))}
            {isLoading && conversation.messages[conversation.messages.length - 1]?.role === 'user' && (
              <div className="flex justify-start mb-4">
                <div className="bg-ai-bubble-dark dark:bg-ai-bubble-dark px-4 py-3 rounded-2xl rounded-tl-sm">
                  <div className="flex items-center gap-2">
                    <RefreshCw className="w-4 h-4 text-accent animate-spin" />
                    <span className="text-gray-400 text-sm">Generating C code...</span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="p-4 border-t border-gray-800 bg-chat-dark">
        <div className="max-w-4xl mx-auto">
          <div className="relative flex items-end gap-2 bg-gray-800 rounded-xl border border-gray-700 focus-within:border-accent transition-colors">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask me to generate C code..."
              disabled={isLoading}
              className="flex-1 bg-transparent text-white px-4 py-3 resize-none focus:outline-none placeholder-gray-500 min-h-[52px] max-h-[200px]"
              rows={1}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={`m-2 p-2 rounded-lg transition-all duration-200 ${
                input.trim() && !isLoading
                  ? 'bg-accent hover:bg-accent-hover text-white'
                  : 'bg-gray-700 text-gray-500 cursor-not-allowed'
              }`}
              title="Send message"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-xs text-gray-600 text-center mt-2">
            Press Enter to send, Shift+Enter for new line
          </p>
        </div>
      </div>
    </div>
  );
}
