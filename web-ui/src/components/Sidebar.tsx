'use client';

import { useState, useRef, useEffect } from 'react';
import {
  MessageSquarePlus,
  Trash2,
  Edit3,
  Check,
  X,
  Settings,
  Sun,
  Moon,
  ChevronDown,
  Code2,
  Sparkles,
  PanelLeftClose,
  MoreVertical,
} from 'lucide-react';
import { Conversation, Settings as SettingsType } from '@/types';

interface SidebarProps {
  conversations: Conversation[];
  activeConversationId: string | null;
  settings: SettingsType;
  sidebarOpen: boolean;
  onNewChat: () => void;
  onSelectConversation: (id: string) => void;
  onDeleteConversation: (id: string) => void;
  onRenameConversation: (id: string, title: string) => void;
  onClearConversation: (id: string) => void;
  onSettingsChange: (settings: SettingsType) => void;
  onToggleSidebar: () => void;
}

const MODELS = [
  { id: 'c-code-llm', name: 'C-Code LLM', description: 'Main C code generator' },
  { id: 'c-code-llm-fast', name: 'C-Code LLM Fast', description: 'Faster, lighter model' },
];

const SYSTEM_PROMPTS = [
  { id: 'default', name: 'Default', description: 'Balanced code generation' },
  { id: 'strict', name: 'Strict C99', description: 'Pure C code only' },
  { id: 'educational', name: 'Educational', description: 'With detailed comments' },
  { id: 'optimized', name: 'Optimized', description: 'Performance focused' },
];

export default function Sidebar({
  conversations,
  activeConversationId,
  settings,
  sidebarOpen,
  onNewChat,
  onSelectConversation,
  onDeleteConversation,
  onRenameConversation,
  onClearConversation,
  onSettingsChange,
  onToggleSidebar,
}: SidebarProps) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editTitle, setEditTitle] = useState('');
  const [showModelDropdown, setShowModelDropdown] = useState(false);
  const [showPromptDropdown, setShowPromptDropdown] = useState(false);
  const [contextMenuId, setContextMenuId] = useState<string | null>(null);
  const editInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editingId && editInputRef.current) {
      editInputRef.current.focus();
    }
  }, [editingId]);

  useEffect(() => {
    const handleClickOutside = () => {
      setShowModelDropdown(false);
      setShowPromptDropdown(false);
      setContextMenuId(null);
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const startEditing = (conv: Conversation) => {
    setEditingId(conv.id);
    setEditTitle(conv.title);
    setContextMenuId(null);
  };

  const saveEdit = () => {
    if (editingId && editTitle.trim()) {
      onRenameConversation(editingId, editTitle.trim());
    }
    setEditingId(null);
    setEditTitle('');
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditTitle('');
  };

  if (!sidebarOpen) {
    return null;
  }

  return (
    <div className="w-[280px] min-w-[280px] h-full bg-sidebar-dark dark:bg-sidebar-dark light:bg-sidebar-light flex flex-col border-r border-gray-800 dark:border-gray-800">
      {/* Header */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
              <Code2 className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-white font-semibold text-sm">C-Code LLM</h1>
              <p className="text-gray-500 text-xs">Studio</p>
            </div>
          </div>
          <button
            onClick={onToggleSidebar}
            className="p-2 text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-colors"
            title="Close sidebar"
          >
            <PanelLeftClose className="w-5 h-5" />
          </button>
        </div>
        
        {/* New Chat Button */}
        <button
          onClick={onNewChat}
          className="w-full flex items-center justify-center gap-2 py-3 px-4 bg-accent hover:bg-accent-hover text-white rounded-lg font-medium transition-all duration-200"
        >
          <MessageSquarePlus className="w-5 h-5" />
          New Chat
        </button>
      </div>

      {/* Conversations List */}
      <div className="flex-1 overflow-y-auto p-2">
        <div className="text-xs text-gray-500 uppercase tracking-wider px-2 py-2">
          Conversations
        </div>
        {conversations.length === 0 ? (
          <div className="text-center text-gray-500 py-8 px-4">
            <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-50" />
            <p className="text-sm">No conversations yet</p>
            <p className="text-xs mt-1">Start a new chat to begin</p>
          </div>
        ) : (
          <div className="space-y-1">
            {conversations.map((conv) => (
              <div
                key={conv.id}
                className={`group relative conversation-item rounded-lg ${
                  activeConversationId === conv.id
                    ? 'bg-gray-800'
                    : ''
                }`}
              >
                {editingId === conv.id ? (
                  <div className="flex items-center gap-1 p-2">
                    <input
                      ref={editInputRef}
                      type="text"
                      value={editTitle}
                      onChange={(e) => setEditTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') saveEdit();
                        if (e.key === 'Escape') cancelEdit();
                      }}
                      className="flex-1 bg-gray-700 text-white text-sm px-2 py-1 rounded border border-gray-600 focus:outline-none focus:border-accent"
                    />
                    <button
                      onClick={saveEdit}
                      className="p-1 text-green-500 hover:bg-gray-700 rounded"
                    >
                      <Check className="w-4 h-4" />
                    </button>
                    <button
                      onClick={cancelEdit}
                      className="p-1 text-red-500 hover:bg-gray-700 rounded"
                    >
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                ) : (
                  <div
                    onClick={() => onSelectConversation(conv.id)}
                    className="flex items-center justify-between p-3 cursor-pointer"
                  >
                    <div className="flex-1 min-w-0">
                      <p className="text-gray-200 text-sm truncate">{conv.title}</p>
                      <p className="text-gray-500 text-xs">
                        {conv.messages.length} messages
                      </p>
                    </div>
                    <div className="relative">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setContextMenuId(contextMenuId === conv.id ? null : conv.id);
                        }}
                        className="p-1 text-gray-500 hover:text-white opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <MoreVertical className="w-4 h-4" />
                      </button>
                      {contextMenuId === conv.id && (
                        <div
                          className="absolute right-0 top-full mt-1 w-36 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 overflow-hidden"
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            onClick={() => startEditing(conv)}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700"
                          >
                            <Edit3 className="w-4 h-4" />
                            Rename
                          </button>
                          <button
                            onClick={() => {
                              onClearConversation(conv.id);
                              setContextMenuId(null);
                            }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-700"
                          >
                            <Trash2 className="w-4 h-4" />
                            Clear
                          </button>
                          <button
                            onClick={() => {
                              onDeleteConversation(conv.id);
                              setContextMenuId(null);
                            }}
                            className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-gray-700"
                          >
                            <Trash2 className="w-4 h-4" />
                            Delete
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Settings Section */}
      <div className="p-3 border-t border-gray-800 space-y-3">
        <div className="flex items-center gap-2 px-2 text-xs text-gray-500 uppercase tracking-wider">
          <Settings className="w-3 h-3" />
          Settings
        </div>

        {/* Model Selector */}
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowModelDropdown(!showModelDropdown);
              setShowPromptDropdown(false);
            }}
            className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg text-sm text-gray-200 hover:bg-gray-700 transition-colors"
          >
            <span>Model: {MODELS.find((m) => m.id === settings.model)?.name}</span>
            <ChevronDown className="w-4 h-4 text-gray-400" />
          </button>
          {showModelDropdown && (
            <div className="absolute bottom-full left-0 w-full mb-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 overflow-hidden">
              {MODELS.map((model) => (
                <button
                  key={model.id}
                  onClick={() => {
                    onSettingsChange({ ...settings, model: model.id });
                    setShowModelDropdown(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-700 transition-colors ${
                    settings.model === model.id ? 'bg-gray-700 text-accent' : 'text-gray-200'
                  }`}
                >
                  <div className="font-medium">{model.name}</div>
                  <div className="text-xs text-gray-500">{model.description}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* System Prompt Selector */}
        <div className="relative">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setShowPromptDropdown(!showPromptDropdown);
              setShowModelDropdown(false);
            }}
            className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg text-sm text-gray-200 hover:bg-gray-700 transition-colors"
          >
            <span>Style: {SYSTEM_PROMPTS.find((p) => p.id === settings.systemPrompt)?.name}</span>
            <ChevronDown className="w-4 h-4 text-gray-400" />
          </button>
          {showPromptDropdown && (
            <div className="absolute bottom-full left-0 w-full mb-1 bg-gray-800 border border-gray-700 rounded-lg shadow-xl z-50 overflow-hidden">
              {SYSTEM_PROMPTS.map((prompt) => (
                <button
                  key={prompt.id}
                  onClick={() => {
                    onSettingsChange({ ...settings, systemPrompt: prompt.id });
                    setShowPromptDropdown(false);
                  }}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-700 transition-colors ${
                    settings.systemPrompt === prompt.id ? 'bg-gray-700 text-accent' : 'text-gray-200'
                  }`}
                >
                  <div className="font-medium">{prompt.name}</div>
                  <div className="text-xs text-gray-500">{prompt.description}</div>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* Theme Toggle */}
        <button
          onClick={() =>
            onSettingsChange({
              ...settings,
              theme: settings.theme === 'dark' ? 'light' : 'dark',
            })
          }
          className="w-full flex items-center justify-between px-3 py-2 bg-gray-800 rounded-lg text-sm text-gray-200 hover:bg-gray-700 transition-colors"
        >
          <span>Theme</span>
          <div className="flex items-center gap-2">
            {settings.theme === 'dark' ? (
              <>
                <Moon className="w-4 h-4" />
                <span className="text-xs">Dark</span>
              </>
            ) : (
              <>
                <Sun className="w-4 h-4" />
                <span className="text-xs">Light</span>
              </>
            )}
          </div>
        </button>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-800 text-center">
        <p className="text-xs text-gray-600">
          Built with ❤️ for C developers
        </p>
      </div>
    </div>
  );
}
