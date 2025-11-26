'use client';

import { useState, useEffect } from 'react';
import Sidebar from '@/components/Sidebar';
import ChatArea from '@/components/ChatArea';
import { Conversation, Message, Settings } from '@/types';
import { v4 as uuidv4 } from 'uuid';

const DEFAULT_SETTINGS: Settings = {
  model: 'c-code-llm',
  systemPrompt: 'default',
  theme: 'dark',
};

const SYSTEM_PROMPTS: Record<string, string> = {
  default: 'You are an expert C programmer. Generate clean, efficient, and well-documented C code only. Always include necessary headers and follow best practices.',
  strict: 'You are a C code generator. Output ONLY valid C code with no explanations. Include all necessary #include directives. Follow strict C99 standard.',
  educational: 'You are a C programming teacher. Generate C code with detailed comments explaining each part. Help users learn C programming concepts.',
  optimized: 'You are a performance-focused C programmer. Generate highly optimized C code with minimal memory usage and maximum efficiency.',
};

export default function Home() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);

  // Load from localStorage on mount
  useEffect(() => {
    const savedConversations = localStorage.getItem('c-code-conversations');
    const savedSettings = localStorage.getItem('c-code-settings');
    
    if (savedConversations) {
      const parsed = JSON.parse(savedConversations);
      setConversations(parsed);
      if (parsed.length > 0) {
        setActiveConversationId(parsed[0].id);
      }
    }
    
    if (savedSettings) {
      setSettings(JSON.parse(savedSettings));
    }
  }, []);

  // Save to localStorage on change
  useEffect(() => {
    localStorage.setItem('c-code-conversations', JSON.stringify(conversations));
  }, [conversations]);

  useEffect(() => {
    localStorage.setItem('c-code-settings', JSON.stringify(settings));
    
    // Apply theme
    if (settings.theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [settings]);

  const activeConversation = conversations.find(c => c.id === activeConversationId);

  const createNewConversation = () => {
    const newConversation: Conversation = {
      id: uuidv4(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    setConversations(prev => [newConversation, ...prev]);
    setActiveConversationId(newConversation.id);
  };

  const deleteConversation = (id: string) => {
    setConversations(prev => prev.filter(c => c.id !== id));
    if (activeConversationId === id) {
      const remaining = conversations.filter(c => c.id !== id);
      setActiveConversationId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const renameConversation = (id: string, newTitle: string) => {
    setConversations(prev =>
      prev.map(c =>
        c.id === id ? { ...c, title: newTitle, updatedAt: new Date().toISOString() } : c
      )
    );
  };

  const clearConversation = (id: string) => {
    setConversations(prev =>
      prev.map(c =>
        c.id === id ? { ...c, messages: [], updatedAt: new Date().toISOString() } : c
      )
    );
  };

  const sendMessage = async (content: string) => {
    if (!activeConversationId || isLoading) return;

    const userMessage: Message = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: new Date().toISOString(),
    };

    // Add user message
    setConversations(prev =>
      prev.map(c => {
        if (c.id === activeConversationId) {
          const isFirstMessage = c.messages.length === 0;
          return {
            ...c,
            messages: [...c.messages, userMessage],
            title: isFirstMessage ? content.slice(0, 50) + (content.length > 50 ? '...' : '') : c.title,
            updatedAt: new Date().toISOString(),
          };
        }
        return c;
      })
    );

    // Create assistant message placeholder
    const assistantMessageId = uuidv4();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date().toISOString(),
    };

    setConversations(prev =>
      prev.map(c =>
        c.id === activeConversationId
          ? { ...c, messages: [...c.messages, assistantMessage] }
          : c
      )
    );

    setIsLoading(true);

    try {
      // Stream response from API
      const response = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: content,
          systemPrompt: SYSTEM_PROMPTS[settings.systemPrompt],
          model: settings.model,
          conversationHistory: activeConversation?.messages || [],
        }),
      });

      if (!response.ok) throw new Error('API request failed');

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (reader) {
        let fullContent = '';
        
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          
          const chunk = decoder.decode(value, { stream: true });
          fullContent += chunk;
          
          // Update message with streamed content
          setConversations(prev =>
            prev.map(c =>
              c.id === activeConversationId
                ? {
                    ...c,
                    messages: c.messages.map(m =>
                      m.id === assistantMessageId
                        ? { ...m, content: fullContent }
                        : m
                    ),
                  }
                : c
            )
          );
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);
      // Update with error message
      setConversations(prev =>
        prev.map(c =>
          c.id === activeConversationId
            ? {
                ...c,
                messages: c.messages.map(m =>
                  m.id === assistantMessageId
                    ? { ...m, content: '```c\n// Error: Failed to generate code. Please try again.\n```' }
                    : m
                ),
              }
            : c
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`flex h-screen ${settings.theme === 'dark' ? 'dark' : ''}`}>
      <Sidebar
        conversations={conversations}
        activeConversationId={activeConversationId}
        settings={settings}
        sidebarOpen={sidebarOpen}
        onNewChat={createNewConversation}
        onSelectConversation={setActiveConversationId}
        onDeleteConversation={deleteConversation}
        onRenameConversation={renameConversation}
        onClearConversation={clearConversation}
        onSettingsChange={setSettings}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
      <ChatArea
        conversation={activeConversation}
        isLoading={isLoading}
        sidebarOpen={sidebarOpen}
        onSendMessage={sendMessage}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
        onNewChat={createNewConversation}
      />
    </div>
  );
}
