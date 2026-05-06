/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { 
  Globe, 
  Search, 
  Languages, 
  Info, 
  ArrowRight, 
  History, 
  Trash2, 
  Loader2,
  Sparkles,
  ChevronRight
} from 'lucide-react';
import { GoogleGenAI } from "@google/genai";

interface DetectionResult {
  prediction: string;
  confidence: number;
  sample: string;
  details: string;
}

interface HistoryItem extends DetectionResult {
  id: string;
  originalText: string;
  timestamp: number;
}

const ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY! });

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Load history from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('lang_detector_history');
    if (saved) {
      try {
        setHistory(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to parse history", e);
      }
    }
  }, []);

  // Save history to localStorage
  useEffect(() => {
    localStorage.setItem('lang_detector_history', JSON.stringify(history));
  }, [history]);

  const handleDetect = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setError(null);

    try {
      const prompt = `Identify the language of the following text. 
      Return ONLY a JSON object with the following fields:
      {
        "prediction": "The detected language name (e.g., English, French, Spanish)",
        "confidence": A float between 0 and 1 representing certainty,
        "sample": "A short excerpt of the text translated to English if it's not English, otherwise same as input",
        "details": "A brief explanation of why this language was chosen (e.g., specific character sets, common stop words)"
      }
      
      Text: "${text.substring(0, 500)}"`;

      const response = await ai.models.generateContent({
        model: "gemini-3-flash-preview",
        contents: prompt
      });

      if (!response.text) throw new Error('Detection failed');

      const data = JSON.parse(response.text.replace(/```json|```/g, ""));
      setResult(data);

      const newItem: HistoryItem = {
        ...data,
        id: crypto.randomUUID(),
        originalText: text,
        timestamp: Date.now(),
      };

      setHistory(prev => [newItem, ...prev.slice(0, 9)]); 
    } catch (err) {
      setError('Could not connect to detection service. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const clearHistory = () => {
    if (window.confirm('Clear all detection history?')) {
      setHistory([]);
    }
  };

  const deleteHistoryItem = (id: string) => {
    setHistory(prev => prev.filter(item => item.id !== id));
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Header */}
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12 max-w-2xl w-full"
      >
        <div className="flex justify-center mb-4">
          <div className="bg-indigo-600 p-3 rounded-2xl shadow-lg shadow-indigo-200">
            <Languages className="w-8 h-8 text-white" />
          </div>
        </div>
        <h1 className="text-4xl font-bold text-gray-900 tracking-tight sm:text-5xl mb-3">
          Language Detector <span className="text-indigo-600">AI</span>
        </h1>
        <p className="text-lg text-gray-600 font-medium">
          Identify over 100 languages instantly with neural-powered precision.
        </p>
      </motion.header>

      <main className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Input & Result Area */}
        <div className="lg:col-span-8 space-y-6">
          <motion.div 
            layout
            className="bg-white rounded-3xl shadow-xl shadow-gray-200/50 border border-gray-100 overflow-hidden"
          >
            <div className="p-6 sm:p-8">
              <div className="flex items-center justify-between mb-4">
                <label htmlFor="text-input" className="text-sm font-semibold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                  <Search className="w-4 h-4" />
                  Analyze Text
                </label>
                <span className="text-xs text-gray-400 font-mono">
                  {text.length} / 500 characters
                </span>
              </div>
              
              <textarea
                id="text-input"
                className="w-full h-40 p-4 text-lg text-gray-800 placeholder-gray-300 border-none focus:ring-0 resize-none bg-gray-100 rounded-2xl transition-all"
                placeholder="Paste text snippets here... e.g. 'Bonjour le monde' or '你好世界'"
                maxLength={500}
                value={text}
                onChange={(e) => setText(e.target.value)}
              />

              <div className="mt-6 flex flex-col sm:flex-row gap-4 items-center justify-between">
                <div className="flex items-center gap-2 text-sm text-gray-500">
                  <Sparkles className="w-4 h-4 text-amber-500" />
                  <span>Uses neural embeddings for high accuracy</span>
                </div>
                
                <button
                  onClick={handleDetect}
                  disabled={loading || !text.trim()}
                  className={`
                    w-full sm:w-auto px-8 py-4 rounded-2xl font-bold text-white transition-all transform
                    flex items-center justify-center gap-3
                    ${loading || !text.trim() 
                      ? 'bg-gray-300 cursor-not-allowed' 
                      : 'bg-indigo-600 hover:bg-indigo-700 hover:scale-[1.02] active:scale-[0.98] shadow-lg shadow-indigo-200'}
                  `}
                >
                  {loading ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <ArrowRight className="w-5 h-5" />
                  )}
                  {loading ? 'Analyzing...' : 'Detect Language'}
                </button>
              </div>

              {error && (
                <motion.p 
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="mt-4 text-red-500 text-sm font-medium bg-red-50 p-3 rounded-lg"
                >
                  {error}
                </motion.p>
              )}
            </div>
          </motion.div>

          <AnimatePresence>
            {result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.95 }}
                className="bg-indigo-600 rounded-3xl shadow-xl shadow-indigo-200 border border-indigo-500 text-white p-8"
              >
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
                  <div className="space-y-4">
                    <div className="flex items-center gap-3">
                      <div className="bg-white/20 p-2 rounded-xl backdrop-blur-sm">
                        <Globe className="w-6 h-6" />
                      </div>
                      <span className="text-white/80 font-semibold tracking-wider text-sm uppercase">Result</span>
                    </div>
                    <h2 className="text-5xl font-black">{result.prediction}</h2>
                    <div className="flex items-center gap-4">
                      <div className="h-2 flex-1 bg-white/20 rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${result.confidence * 100}%` }}
                          transition={{ duration: 0.8, ease: "easeOut" }}
                          className="h-full bg-white shadow-[0_0_10px_white/50]"
                        />
                      </div>
                      <span className="font-mono text-xl font-bold">
                        {(result.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="bg-white/10 backdrop-blur-md rounded-2xl p-6 space-y-4 border border-white/10">
                    <div className="flex items-start gap-3">
                      <Info className="w-5 h-5 mt-1 shrink-0 opacity-70" />
                      <div>
                        <h4 className="font-bold mb-1 opacity-90 italic">Why this choice?</h4>
                        <p className="text-sm text-white/80 leading-relaxed">
                          {result.details}
                        </p>
                      </div>
                    </div>
                    {result.prediction !== 'English' && (
                      <div className="pt-4 border-t border-white/10">
                        <h4 className="text-xs font-bold uppercase tracking-widest mb-2 opacity-50">Translated Preview</h4>
                        <p className="text-lg font-medium italic">"{result.sample}"</p>
                      </div>
                    )}
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* History Sidebar */}
        <div className="lg:col-span-4 space-y-6">
          <div className="bg-white rounded-3xl shadow-lg shadow-gray-100 border border-gray-100 p-6">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-lg font-bold text-gray-900 flex items-center gap-2">
                <History className="w-5 h-5 text-gray-400" />
                History
              </h3>
              {history.length > 0 && (
                <button 
                  onClick={clearHistory}
                  className="text-xs text-red-500 font-bold hover:bg-red-50 px-2 py-1 rounded-lg transition-colors"
                >
                  Clear
                </button>
              )}
            </div>

            <div className="space-y-4">
              <AnimatePresence mode='popLayout'>
                {history.length === 0 ? (
                  <div className="text-center py-8">
                    <p className="text-sm text-gray-400 italic">No recent detections</p>
                  </div>
                ) : (
                  history.map((item) => (
                    <motion.div
                      key={item.id}
                      layout
                      initial={{ opacity: 0, x: 20 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -20 }}
                      className="group relative bg-gray-50 rounded-2xl p-4 border border-transparent hover:border-indigo-100 hover:bg-white hover:shadow-md transition-all cursor-pointer"
                      onClick={() => {
                        setText(item.originalText);
                        setResult(item);
                        window.scrollTo({ top: 0, behavior: 'smooth' });
                      }}
                    >
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteHistoryItem(item.id);
                        }}
                        className="absolute -top-2 -right-2 bg-white text-gray-400 hover:text-red-500 p-1.5 rounded-full shadow-sm opacity-0 group-hover:opacity-100 transition-opacity border border-gray-100"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                      
                      <div className="flex justify-between items-start mb-2">
                        <span className="text-xs font-bold text-indigo-600 bg-indigo-50 px-2 py-0.5 rounded-md">
                          {item.prediction}
                        </span>
                        <span className="text-[10px] text-gray-400 font-mono">
                          {new Date(item.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </span>
                      </div>
                      <p className="text-sm text-gray-600 line-clamp-2 italic mb-2">
                        "{item.originalText}"
                      </p>
                      <div className="flex items-center gap-1 text-[10px] font-bold text-gray-400 uppercase tracking-tighter">
                        Confidence
                        <ChevronRight className="w-3 h-3" />
                        <span className="text-gray-900">{(item.confidence * 100).toFixed(0)}%</span>
                      </div>
                    </motion.div>
                  ))
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Quick Info Card */}
          <div className="bg-indigo-900 rounded-3xl p-6 text-white overflow-hidden relative">
            <Sparkles className="absolute -bottom-4 -right-4 w-24 h-24 text-white/10 rotate-12" />
            <h4 className="text-xl font-bold mb-4 relative z-10">Did you know?</h4>
            <p className="text-indigo-100 text-sm leading-relaxed relative z-10">
              There are over 7,000 languages spoken globally. Our AI looks at character frequencies and n-gram patterns to distinguish closely related ones like Hindi and Marathi.
            </p>
          </div>
        </div>
      </main>

      <footer className="mt-16 text-center text-gray-400 text-sm">
        <p>© 2026 Language Detector AI • Powered by Scikit-Learn & Gemini</p>
      </footer>
    </div>
  );
}
