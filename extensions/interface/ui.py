import React, { useState, useEffect, useRef } from 'react';
import { Send, Settings, Brain, Download, Upload, Info, Moon, Menu, ChevronDown, 
         Shield, HelpCircle, FileText, Award, Code } from 'lucide-react';

const SullyUI = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [cognitiveMode, setCognitiveMode] = useState('emergent');
  const [settingsOpen, setSettingsOpen] = useState(false);

  const [userConsent, setUserConsent] = useState(null);
  const [consentModalOpen, setConsentModalOpen] = useState(true);

  const messagesEndRef = useRef(null);

  const [aboutMenuOpen, setAboutMenuOpen] = useState(false);
  const [gamesMenuOpen, setGamesMenuOpen] = useState(false);
  const [permissionsMenuOpen, setPermissionsMenuOpen] = useState(false);

  const aboutMenuRef = useRef(null);
  const gamesMenuRef = useRef(null);
  const permissionsMenuRef = useRef(null);

  const cognitiveOptions = [
    'emergent', 'analytical', 'creative', 'critical', 
    'ethereal', 'humorous', 'professional', 'casual'
  ];

  useEffect(() => {
    const stored = localStorage.getItem("sullyUserConsent");
    if (stored !== null) {
      setUserConsent(stored === "true");
      setConsentModalOpen(false);
    }
  }, []);

  useEffect(() => {
    function handleClickOutside(event) {
      if (aboutMenuRef.current && !aboutMenuRef.current.contains(event.target)) {
        setAboutMenuOpen(false);
      }
      if (gamesMenuRef.current && !gamesMenuRef.current.contains(event.target)) {
        setGamesMenuOpen(false);
      }
      if (permissionsMenuRef.current && !permissionsMenuRef.current.contains(event.target)) {
        setPermissionsMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = () => {
    if (inputValue.trim() === '') return;

    setMessages(prev => [...prev, { 
      id: prev.length + 1, 
      content: inputValue, 
      sender: 'user' 
    }]);

    setTimeout(() => {
      setMessages(prev => [...prev, { 
        id: prev.length + 1, 
        content: `I'm processing your inquiry about "${inputValue}" using my ${cognitiveMode} cognitive mode.`, 
        sender: 'sully' 
      }]);
    }, 1000);

    setInputValue('');
  };

  const handleConsent = (consent) => {
    setUserConsent(consent);
    localStorage.setItem("sullyUserConsent", consent);
    setConsentModalOpen(false);
  };

  return (
    <div className="relative w-full h-screen overflow-hidden bg-black">
      {consentModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50">
          <div className="bg-gray-900 border border-purple-600 p-6 rounded-xl shadow-lg max-w-md text-white space-y-4">
            <h2 className="text-xl font-semibold">Allow Sully to Remember You?</h2>
            <p>Sully can store your context, preferences, and memory across sessions to continue conversations seamlessly. Do you want to enable this?</p>
            <div className="flex justify-end space-x-4">
              <button onClick={() => handleConsent(false)} className="px-4 py-2 bg-gray-700 rounded hover:bg-gray-600">No thanks</button>
              <button onClick={() => handleConsent(true)} className="px-4 py-2 bg-purple-600 rounded hover:bg-purple-500">Yes, remember me</button>
            </div>
          </div>
        </div>
      )}

      {/* Existing UI continues here... */}
      {/* Keep your header, chat window, input area, and all prior structure below */}
    </div>
  );
};

export default SullyUI;