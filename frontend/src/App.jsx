import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { 
  Shield, AlertTriangle, Search, Loader2, CheckCircle2, Lock, Zap, Globe, 
  Menu, X, History, LayoutDashboard, Trash2, Sun, Moon, 
  FileText, RefreshCw, Copy, Edit3, Repeat, Sparkles, HelpCircle, User, Bot, LogIn, ArrowRight, Mail, Key, UserPlus
} from 'lucide-react';

function App() {
  // --- STATE ---
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState({ status: "Checking...", device: "" });
  
  // Paraphraser State
  const [paraInput, setParaInput] = useState('');
  const [paraSentences, setParaSentences] = useState([]); 
  const [paraLoading, setParaLoading] = useState(false);
  const [tone, setTone] = useState('Standard'); 

  // Interactive State
  const [activeWord, setActiveWord] = useState({ idx: -1, sentIdx: -1 }); 
  const [activeSentence, setActiveSentence] = useState(-1); 
  const [suggestions, setSuggestions] = useState([]); 
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  
  // Popups Positioning State
  const [popupPosition, setPopupPosition] = useState({ top: 0, left: 0, placement: 'bottom' });

  // Navigation & History
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [currentView, setCurrentView] = useState('scan'); 
  const [history, setHistory] = useState([]);
  const [themeState, setThemeState] = useState('dark');
  const isDark = themeState === 'dark';

  // --- AUTHENTICATION STATE ---
  const [isLoginView, setIsLoginView] = useState(true); 
  const [currentUser, setCurrentUser] = useState(null); 
  
  // Form Inputs
  const [authName, setAuthName] = useState('');
  const [authEmail, setAuthEmail] = useState('');
  const [authPassword, setAuthPassword] = useState('');

  useEffect(() => {
    axios.get('http://127.0.0.1:8000/').then(res => setBackendStatus(res.data)).catch(() => {});
    
    // Theme Loading
    const savedTheme = localStorage.getItem('integrityTheme');
    if (savedTheme) setThemeState(savedTheme);

    // Check for persisted session (Auto-login logic optional, strict login preferred for security demo)
    // For this demo, we start logged out to show the flow.
  }, []);

  // --- HELPER: HISTORY MANAGER ---
  const saveToHistory = (newItem) => {
    if (!currentUser) return; // Don't save if not logged in

    const updatedHistory = [newItem, ...history];
    setHistory(updatedHistory);
    
    // SAVE TO SPECIFIC USER KEY: 'history_user@email.com'
    localStorage.setItem(`integrityHistory_${currentUser.email}`, JSON.stringify(updatedHistory));
  };

  const loadUserHistory = (email) => {
    const userHistory = localStorage.getItem(`integrityHistory_${email}`);
    if (userHistory) {
      setHistory(JSON.parse(userHistory));
    } else {
      setHistory([]);
    }
  };

  // --- HELPER: SMART POSITIONING ---
  const calculatePosition = (event) => {
    const rect = event.target.getBoundingClientRect();
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceRight = window.innerWidth - rect.right;
    const placement = spaceBelow < 250 ? 'top' : 'bottom';
    
    let top = placement === 'bottom' ? rect.bottom + window.scrollY + 5 : rect.top + window.scrollY - 5;
    let left = rect.left + window.scrollX;
    if (spaceRight < 200) left = window.innerWidth - 220; 

    setPopupPosition({ top, left, placement });
  };

  // --- ACTIONS ---
  const handleAnalyze = async () => {
    if (!text.trim()) return;
    setLoading(true); setResult(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/analyze', { content: text });
      const scanResult = response.data;
      setResult(scanResult);
      
      const newEntry = {
        id: Date.now(), type: 'scan', textSnippet: text.substring(0, 50)+"...", fullText: text,
        prediction: scanResult.prediction, confidence: scanResult.confidence, risk: scanResult.risk_level,
        date: new Date().toLocaleDateString(), time: new Date().toLocaleTimeString()
      };
      
      saveToHistory(newEntry);

    } catch (e) { alert("Backend Error"); }
    setLoading(false);
  };

  const handleParaphrase = async () => {
    if (!paraInput.trim()) return;
    setParaLoading(true); setParaSentences([]);
    try {
      const response = await axios.post('http://127.0.0.1:8000/paraphrase', { content: paraInput, tone: tone });
      const rawText = response.data.paraphrased;
      const splitSentences = rawText.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [rawText];
      setParaSentences(splitSentences.map(s => s.trim()));
      
      const newEntry = {
        id: Date.now(), type: 'paraphrase', original: paraInput, result: rawText, tone: tone,
        date: new Date().toLocaleDateString(), time: new Date().toLocaleTimeString()
      };
      
      saveToHistory(newEntry);

    } catch (e) { alert("Backend Error"); }
    setParaLoading(false);
  };

  // Word Click
  const handleWordClick = async (e, word, sentIdx, wordIdx) => {
    e.stopPropagation(); calculatePosition(e);
    const cleanWord = word.replace(/[^\w\s]/gi, '');
    if (!cleanWord) return;

    setActiveWord({ idx: wordIdx, sentIdx: sentIdx });
    setActiveSentence(-1); setLoadingSuggestions(true); setSuggestions([]);

    try {
      const res = await axios.post('http://127.0.0.1:8000/synonyms', { word: cleanWord });
      setSuggestions(res.data.synonyms);
    } catch (e) {}
    setLoadingSuggestions(false);
  };

  // Sentence Click
  const handleSentenceClick = async (e, sentence, sentIdx) => {
    e.stopPropagation(); calculatePosition(e);
    setActiveSentence(sentIdx);
    setActiveWord({ idx: -1, sentIdx: -1 }); setLoadingSuggestions(true); setSuggestions([]);

    try {
      const res = await axios.post('http://127.0.0.1:8000/rewrite_sentence', { sentence: sentence, tone: tone });
      setSuggestions(res.data.variants);
    } catch (e) {}
    setLoadingSuggestions(false);
  };

  // Replacements
  const replaceWord = (sentIdx, wordIdx, newWord) => {
    const newSentences = [...paraSentences];
    const words = newSentences[sentIdx].split(' ');
    const original = words[wordIdx];
    const punctuation = original.match(/[.,!?;:]$/) ? original.slice(-1) : '';
    if (original[0] === original[0].toUpperCase()) newWord = newWord.charAt(0).toUpperCase() + newWord.slice(1);
    words[wordIdx] = newWord + punctuation;
    newSentences[sentIdx] = words.join(' ');
    setParaSentences(newSentences); setActiveWord({ idx: -1, sentIdx: -1 });
  };

  const replaceSentence = (sentIdx, newSentence) => {
    const newSentences = [...paraSentences];
    newSentences[sentIdx] = newSentence;
    setParaSentences(newSentences); setActiveSentence(-1);
  };

  // --- AUTH LOGIC (MOCK DATABASE) ---
  const handleAuth = (e) => {
    e.preventDefault();
    
    // Get existing users from "Database" (LocalStorage)
    const storedUsers = JSON.parse(localStorage.getItem('integrity_users') || '[]');

    if (isLoginView) {
      // --- LOGIN LOGIC ---
      const userFound = storedUsers.find(u => u.email === authEmail && u.password === authPassword);
      
      if (userFound) {
        setCurrentUser(userFound);
        loadUserHistory(userFound.email); // LOAD THIS USER'S HISTORY
        setCurrentView('scan');
        setAuthEmail(''); setAuthPassword(''); // Clear form
      } else {
        alert("Invalid credentials! Please check email or password.");
      }

    } else {
      // --- SIGNUP LOGIC ---
      if (storedUsers.find(u => u.email === authEmail)) {
        alert("User already exists!");
        return;
      }

      const newUser = { name: authName, email: authEmail, password: authPassword };
      storedUsers.push(newUser);
      
      // Save updated DB
      localStorage.setItem('integrity_users', JSON.stringify(storedUsers));
      
      alert("Account Created! Please Sign In.");
      setIsLoginView(true); // Switch to login
      setAuthName(''); setAuthPassword(''); // Keep email for convenience
    }
  };

  const handleLogout = () => {
    setCurrentUser(null);
    setHistory([]); // CLEAR HISTORY FROM VIEW
    setIsLoginView(true);
    setAuthEmail('');
    setAuthPassword('');
  };

  // --- HELPERS FOR STYLING ---
  const getRiskColor = (risk) => {
    if (risk === 'High') return { gradient: 'from-red-500/50 to-orange-500/50', bg: 'bg-red-500/20', text: 'text-red-500', badgeText: 'text-red-600', icon: <AlertTriangle size={48} /> };
    if (risk === 'Medium') return { gradient: 'from-yellow-500/50 to-amber-500/50', bg: 'bg-yellow-500/20', text: 'text-yellow-500', badgeText: 'text-yellow-600', icon: <HelpCircle size={48} /> };
    return { gradient: 'from-emerald-500/50 to-teal-500/50', bg: 'bg-emerald-500/20', text: 'text-emerald-500', badgeText: 'text-emerald-600', icon: <CheckCircle2 size={48} /> };
  };

  const resetApp = () => { setText(''); setResult(null); setCurrentView('scan'); };
  const toggleTheme = () => { const t = isDark ? 'light' : 'dark'; setThemeState(t); localStorage.setItem('integrityTheme', t); };
  const clearHistory = () => { 
    if (confirm("Clear logs?")) { 
      setHistory([]); 
      // Clear storage for current user only
      if(currentUser) localStorage.removeItem(`integrityHistory_${currentUser.email}`);
    } 
  };
  const restoreHistoryItem = (item) => {
    if (item.type === 'scan') { setText(item.fullText); setResult({ prediction: item.prediction, confidence: item.confidence, risk_level: item.risk }); setCurrentView('scan'); }
    else { setParaInput(item.original); setParaSentences(item.result.match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [item.result]); setCurrentView('paraphrase'); }
    setIsSidebarOpen(false);
  };

  return (
    <div className={`relative min-h-screen font-sans transition-colors duration-500 ${isDark ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'}`} onClick={() => { setActiveWord({ idx: -1, sentIdx: -1 }); setActiveSentence(-1); }}>
      
      {/* BACKGROUND */}
      <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden">
        {isDark ? <div className="absolute top-0 left-0 w-full h-full bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black"></div> : <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-blue-50 to-white"></div>}
        <div className="absolute top-0 left-0 w-full h-full opacity-20 bg-[url('https://grainy-gradients.vercel.app/noise.svg')]"></div>
      </div>

      {/* SIDEBAR */}
      {isSidebarOpen && <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-40" onClick={(e) => {e.stopPropagation(); setIsSidebarOpen(false)}}></div>}
      <div className={`fixed top-0 left-0 h-full w-72 z-50 transform transition-transform duration-300 border-r backdrop-blur-xl ${isDark ? 'bg-slate-900/95 border-white/10' : 'bg-white/95 border-slate-200'} ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`} onClick={(e) => e.stopPropagation()}>
        <div className="p-6 flex flex-col h-full">
          <div className="flex items-center justify-between mb-10"><span className={`font-bold text-xl ${isDark ? 'text-white' : 'text-slate-900'}`}>IntegrityAI</span><button onClick={() => setIsSidebarOpen(false)}><X size={24} className="text-slate-500"/></button></div>
          <nav className="flex-grow space-y-2">
            <button onClick={() => {setCurrentView('scan'); setIsSidebarOpen(false)}} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${currentView === 'scan' ? 'bg-blue-600 text-white' : isDark ? 'text-slate-400 hover:bg-white/5' : 'text-slate-500 hover:bg-slate-100'}`}><LayoutDashboard size={20}/>Scanner</button>
            <button onClick={() => {setCurrentView('paraphrase'); setIsSidebarOpen(false)}} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${currentView === 'paraphrase' ? 'bg-blue-600 text-white' : isDark ? 'text-slate-400 hover:bg-white/5' : 'text-slate-500 hover:bg-slate-100'}`}><RefreshCw size={20}/>Paraphraser</button>
            <button onClick={() => {setCurrentView('history'); setIsSidebarOpen(false)}} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl font-medium ${currentView === 'history' ? 'bg-blue-600 text-white' : isDark ? 'text-slate-400 hover:bg-white/5' : 'text-slate-500 hover:bg-slate-100'}`}><History size={20}/>History</button>
          </nav>
          <div className="mt-auto pt-6 border-t border-white/10"><button onClick={toggleTheme} className={`w-full flex items-center justify-between px-4 py-3 rounded-xl border ${isDark ? 'bg-slate-800 border-slate-700' : 'bg-slate-100 border-slate-200'}`}><div className="flex items-center gap-3">{isDark ? <Moon size={18}/> : <Sun size={18}/>}<span className="text-sm font-bold">{isDark?"Dark":"Light"} Mode</span></div></button></div>
        </div>
      </div>

      {/* NAVBAR */}
      <nav className={`border-b sticky top-0 z-30 backdrop-blur-md ${isDark ? 'border-white/5 bg-slate-950/50' : 'border-slate-200 bg-white/70'}`}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between relative">
          <div className="flex items-center gap-4">
            <button onClick={(e) => {e.stopPropagation(); setIsSidebarOpen(true)}} className="p-2 text-slate-500"><Menu size={28}/></button>
            <div className={`hidden md:flex px-3 py-1 rounded-full border items-center gap-2 text-xs font-bold ${backendStatus.status === "Ready" ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border-red-500/20 text-red-400"}`}><div className={`w-2 h-2 rounded-full ${backendStatus.status==="Ready"?"bg-emerald-500 animate-pulse":"bg-red-500"}`}></div>{backendStatus.status==="Ready"?"ONLINE":"OFFLINE"}</div>
          </div>
          
          <div onClick={resetApp} className="absolute left-1/2 transform -translate-x-1/2 flex items-center gap-2 cursor-pointer group select-none">
            <div className="bg-blue-500/10 p-1.5 rounded-lg border border-blue-500/20 hidden sm:block transition-transform group-hover:scale-110"><Shield size={18} className="text-blue-400 drop-shadow-[0_0_8px_rgba(59,130,246,0.6)]" /></div>
            <div className="text-2xl font-black tracking-tighter"><span className={`drop-shadow-[0_0_15px_rgba(255,255,255,0.6)] transition-colors ${isDark ? 'text-white' : 'text-slate-900'}`}>Integrity</span><span className={isDark ? 'text-slate-700' : 'text-slate-400'}>AI</span></div>
          </div>

          <div className="flex items-center gap-4">
            {currentUser ? (
              <div className="flex items-center gap-2 cursor-pointer group" onClick={() => setCurrentView('auth')}>
                <div className="text-right hidden sm:block">
                  <div className={`text-xs font-bold ${isDark?'text-white':'text-slate-900'}`}>{currentUser.name}</div>
                  <div className="text-[10px] text-slate-500">{currentUser.email}</div>
                </div>
                <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-blue-600 to-indigo-600 flex items-center justify-center text-white font-bold shadow-lg ring-2 ring-transparent group-hover:ring-blue-500 transition-all">
                  {currentUser.name[0].toUpperCase()}
                </div>
              </div>
            ) : (
              <button onClick={() => setCurrentView('auth')} className={`flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-bold transition-all ${isDark ? 'bg-white text-black hover:bg-slate-200' : 'bg-slate-900 text-white hover:bg-slate-700'} shadow-lg`}><LogIn size={16} /><span className="hidden sm:inline">Login</span></button>
            )}
          </div>
        </div>
      </nav>

      {/* POPUPS (Word/Sentence) */}
      {activeWord.idx !== -1 && (
        <div className="fixed z-[100] bg-white rounded-xl shadow-2xl overflow-hidden text-sm border border-slate-200 animate-in fade-in zoom-in-95 duration-100" style={{ top: popupPosition.placement === 'bottom' ? popupPosition.top : 'auto', bottom: popupPosition.placement === 'top' ? (window.innerHeight - popupPosition.top + 10) : 'auto', left: popupPosition.left, width: '200px' }} onClick={(e) => e.stopPropagation()}>
          <div className="bg-slate-100 px-3 py-2 text-xs font-bold text-slate-500 border-b">Synonyms</div>
          <div className="max-h-56 overflow-y-auto">
            {loadingSuggestions ? <div className="p-3 text-center"><Loader2 className="animate-spin inline text-indigo-500"/></div> : suggestions.length > 0 ? suggestions.map((syn, i) => (<button key={i} onClick={() => replaceWord(activeWord.sentIdx, activeWord.idx, syn)} className="w-full text-left px-4 py-2 hover:bg-indigo-50 text-slate-800 border-b border-slate-50 last:border-0 hover:text-indigo-600 transition-colors">{syn}</button>)) : <div className="p-2 text-center text-slate-400 text-xs">No synonyms found</div>}
          </div>
        </div>
      )}
      {activeSentence !== -1 && (
         <div className="fixed z-[100] bg-white rounded-xl shadow-2xl overflow-hidden text-sm border border-slate-200 animate-in fade-in zoom-in-95 duration-100" style={{ top: popupPosition.placement === 'bottom' ? popupPosition.top : 'auto', bottom: popupPosition.placement === 'top' ? (window.innerHeight - popupPosition.top + 10) : 'auto', left: Math.max(10, Math.min(popupPosition.left - 150, window.innerWidth - 370)), width: '350px' }} onClick={(e) => e.stopPropagation()}>
            <div className="bg-slate-100 px-3 py-2 text-xs font-bold text-slate-500 uppercase border-b flex justify-between"><span>Sentence Variations</span><button onClick={()=>setActiveSentence(-1)} className="hover:text-red-500"><X size={14}/></button></div>
            <div className="max-h-72 overflow-y-auto">{loadingSuggestions ? <div className="p-6 text-center"><Loader2 className="animate-spin mx-auto text-indigo-500"/></div> : suggestions.length > 0 ? suggestions.map((varSent, i) => (<button key={i} onClick={() => replaceSentence(activeSentence, varSent)} className="w-full text-left px-4 py-3 hover:bg-indigo-50 text-slate-700 font-medium transition-colors border-b border-slate-50 last:border-0 leading-relaxed hover:text-indigo-700">{varSent}</button>)) : <div className="p-4 text-center text-slate-400 italic">No variations found.</div>}</div>
         </div>
      )}

      <main className="flex-grow p-6 w-full max-w-7xl mx-auto flex flex-col items-center">
        
        {/* VIEW: AUTHENTICATION */}
        {currentView === 'auth' && (
          <div className="w-full max-w-md animate-in fade-in slide-in-from-bottom-4 duration-500 mt-12">
             <div className={`p-8 rounded-3xl border shadow-2xl backdrop-blur-xl ${isDark ? 'bg-slate-900/80 border-white/10' : 'bg-white/90 border-slate-200'}`}>
                <div className="text-center mb-8">
                   <div className="inline-flex p-4 rounded-2xl bg-gradient-to-tr from-blue-50 to-indigo-500 text-white mb-4 shadow-lg shadow-blue-500/30"><User size={32} /></div>
                   <h2 className={`text-3xl font-black tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>{currentUser ? "My Profile" : (isLoginView ? "Welcome Back" : "Join IntegrityAI")}</h2>
                   <p className="text-slate-500 mt-2">{currentUser ? "Manage your account details" : (isLoginView ? "Access your history and settings" : "Create a new account to save history")}</p>
                </div>

                {currentUser ? (
                  <div className="space-y-6">
                     <div className={`p-6 rounded-2xl border flex flex-col items-center gap-3 ${isDark ? 'bg-slate-950/50 border-white/10' : 'bg-slate-50 border-slate-200'}`}>
                        <div className="w-20 h-20 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center text-white font-bold text-4xl shadow-xl">{currentUser.name[0].toUpperCase()}</div>
                        <div className="text-center">
                           <div className={`text-xl font-bold ${isDark ? 'text-white' : 'text-slate-900'}`}>{currentUser.name}</div>
                           <div className="text-sm text-slate-500">{currentUser.email}</div>
                        </div>
                     </div>
                     <div className="grid grid-cols-2 gap-4">
                        <div className={`p-4 rounded-xl text-center border ${isDark ? 'bg-slate-950 border-white/5' : 'bg-white border-slate-200'}`}>
                           <div className="text-2xl font-black text-indigo-500">{history.filter(h => h.type==='scan').length}</div>
                           <div className="text-xs font-bold uppercase text-slate-500">Scans</div>
                        </div>
                        <div className={`p-4 rounded-xl text-center border ${isDark ? 'bg-slate-950 border-white/5' : 'bg-white border-slate-200'}`}>
                           <div className="text-2xl font-black text-emerald-500">{history.filter(h => h.type==='paraphrase').length}</div>
                           <div className="text-xs font-bold uppercase text-slate-500">Rewrites</div>
                        </div>
                     </div>
                     <button onClick={handleLogout} className="w-full py-4 rounded-xl font-bold bg-red-500/10 text-red-500 border border-red-500/20 hover:bg-red-500/20 transition-colors flex items-center justify-center gap-2"><LogIn className="rotate-180" size={20}/> Sign Out</button>
                  </div>
                ) : (
                  <form onSubmit={handleAuth} className="space-y-4">
                     {!isLoginView && (
                       <div className="space-y-2 animate-in fade-in slide-in-from-top-2">
                          <label className="text-xs font-bold uppercase text-slate-500 ml-1">Full Name</label>
                          <div className={`flex items-center px-4 py-3 rounded-xl border focus-within:ring-2 focus-within:ring-blue-500 transition-all ${isDark ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-slate-200'}`}>
                             <User size={18} className="text-slate-400 mr-3" />
                             <input type="text" required value={authName} onChange={e => setAuthName(e.target.value)} placeholder="John Doe" className="bg-transparent w-full focus:outline-none" />
                          </div>
                       </div>
                     )}
                     <div className="space-y-2">
                        <label className="text-xs font-bold uppercase text-slate-500 ml-1">Email Address</label>
                        <div className={`flex items-center px-4 py-3 rounded-xl border focus-within:ring-2 focus-within:ring-blue-500 transition-all ${isDark ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-slate-200'}`}>
                           <Mail size={18} className="text-slate-400 mr-3" />
                           <input type="email" required value={authEmail} onChange={e => setAuthEmail(e.target.value)} placeholder="name@example.com" className="bg-transparent w-full focus:outline-none" />
                        </div>
                     </div>
                     <div className="space-y-2">
                        <label className="text-xs font-bold uppercase text-slate-500 ml-1">Password</label>
                        <div className={`flex items-center px-4 py-3 rounded-xl border focus-within:ring-2 focus-within:ring-blue-500 transition-all ${isDark ? 'bg-slate-950 border-white/10' : 'bg-slate-50 border-slate-200'}`}>
                           <Key size={18} className="text-slate-400 mr-3" />
                           <input type="password" required value={authPassword} onChange={e => setAuthPassword(e.target.value)} placeholder="••••••••" className="bg-transparent w-full focus:outline-none" />
                        </div>
                     </div>
                     
                     <button type="submit" className="w-full py-4 rounded-xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 text-white shadow-lg shadow-blue-500/25 hover:shadow-blue-500/40 transition-all transform active:scale-[0.98] flex justify-center items-center gap-2">
                        {isLoginView ? "Sign In" : "Create Account"} <ArrowRight size={18} />
                     </button>

                     <div className="text-center mt-6">
                        <span className="text-slate-500 text-sm">{isLoginView ? "New here?" : "Already have an account?"}</span>
                        <button type="button" onClick={() => setIsLoginView(!isLoginView)} className="ml-2 text-sm font-bold text-blue-500 hover:underline">
                           {isLoginView ? "Create Account" : "Log In"}
                        </button>
                     </div>
                  </form>
                )}
             </div>
          </div>
        )}

        {/* VIEW: SCANNER */}
        {currentView === 'scan' && (
          <div className="w-full flex flex-col items-center animate-fade-in mt-8">
            <div className="text-center max-w-3xl space-y-6 mb-12"><h1 className={`text-5xl md:text-7xl font-black tracking-tighter leading-tight bg-clip-text text-transparent ${isDark ? 'bg-gradient-to-b from-white to-slate-500' : 'bg-gradient-to-b from-slate-900 to-slate-500'}`}>Detect Artificial Patterns <span className="text-blue-500">Instantly.</span></h1></div>
            <div className="w-full max-w-4xl relative group">
               <div className="absolute -inset-1 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-3xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
               <div className={`relative backdrop-blur-xl border rounded-3xl p-2 shadow-2xl ${isDark ? 'bg-slate-900/80 border-white/10' : 'bg-white/80 border-slate-200'}`}>
                 <textarea className={`w-full h-64 bg-transparent p-6 text-lg focus:outline-none resize-none font-medium leading-relaxed rounded-2xl ${isDark ? 'text-slate-200 placeholder-slate-600' : 'text-slate-800 placeholder-slate-400'}`} placeholder="Paste suspect text..." value={text} onChange={(e) => setText(e.target.value.slice(0, 2000))} />
                 <div className={`flex justify-between items-center px-6 py-4 border-t rounded-b-2xl ${isDark ? 'border-white/5 bg-slate-950/30' : 'border-slate-100 bg-slate-50/50'}`}>
                   <span className="text-xs font-mono text-slate-500 uppercase">{text.length} / 2000 chars</span>
                   <button onClick={handleAnalyze} disabled={loading || !text} className="bg-blue-600 hover:bg-blue-500 text-white px-8 py-3 rounded-xl font-bold flex gap-2 items-center">{loading ? <Loader2 className="animate-spin"/> : <Search/>}{loading ? "Scanning..." : "Analyze"}</button>
                 </div>
               </div>
            </div>
            
            {result && (() => {
               const style = getRiskColor(result.risk_level);
               const isAi = ['High', 'Medium'].includes(result.risk_level);
               const aiVal = isAi ? result.confidence : (100 - result.confidence).toFixed(1);
               const humanVal = isAi ? (100 - result.confidence).toFixed(1) : result.confidence;
               const showAi = aiVal > 15;
               const showHuman = humanVal > 15;

               return (
                 <div className="w-full max-w-4xl mt-10 animate-in fade-in slide-in-from-bottom-6">
                   <div className={`p-1 rounded-3xl bg-gradient-to-br ${style.gradient}`}>
                     <div className={`backdrop-blur-2xl rounded-[22px] p-8 ${isDark ? 'bg-slate-950/90 border-white/10' : 'bg-white/90 border-white/40'}`}>
                       <div className="flex flex-col md:flex-row items-center justify-between gap-10">
                         <div className="flex items-center gap-6"><div className={`p-4 rounded-2xl ${style.bg} ${style.text}`}>{style.icon}</div><div><h3 className={`text-3xl font-black ${isDark ? 'text-white' : 'text-slate-900'}`}>{result.prediction}</h3><span className={`text-xs font-bold uppercase tracking-widest px-2 py-0.5 rounded ${style.bg} ${style.badgeText}`}>Risk: {result.risk_level}</span></div></div>
                         <div className="text-right"><div className={`text-7xl font-black ${style.text}`}>{result.confidence}%</div><p className="text-slate-500 text-xs font-bold uppercase tracking-widest">Confidence Score</p></div>
                       </div>
                       <div className={`h-px w-full my-8 ${isDark ? 'bg-white/10' : 'bg-slate-200'}`}></div>
                       <div className="flex flex-col gap-4">
                          <div className="flex justify-between text-sm font-bold uppercase tracking-wider">
                             {showAi ? <div className="flex items-center gap-2 text-indigo-400 animate-in fade-in slide-in-from-left-2"><Bot size={16}/> AI Probability: {aiVal}%</div> : <div></div>}
                             {showHuman ? <div className="flex items-center gap-2 text-emerald-500 animate-in fade-in slide-in-from-right-2"><User size={16}/> Human Probability: {humanVal}%</div> : <div></div>}
                          </div>
                          <div className="h-4 w-full bg-slate-200/20 rounded-full overflow-hidden flex shadow-inner">
                             {showAi && <div className="h-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)] transition-all duration-1000 ease-out" style={{ width: showHuman ? `${aiVal}%` : '100%' }}></div>}
                             {showHuman && <div className="h-full bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)] transition-all duration-1000 ease-out" style={{ width: showAi ? `${humanVal}%` : '100%' }}></div>}
                          </div>
                       </div>
                     </div>
                   </div>
                 </div>
               );
            })()}
            
            <div className="grid md:grid-cols-3 gap-6 w-full max-w-5xl mt-24 mb-12">
               {[ { icon: Lock, title: "Local Privacy", desc: "Data never leaves your local GPU." }, { icon: Zap, title: "RoBERTa Engine", desc: "Powered by fine-tuned models." }, { icon: Globe, title: "Offline Capable", desc: "No internet required." } ].map((f, i) => (<div key={i} className={`p-6 rounded-2xl border ${isDark ? 'bg-slate-900/50 border-white/5' : 'bg-white border-slate-200'}`}><f.icon className="text-blue-500 mb-4" size={24} /><h4 className={`font-bold mb-2 ${isDark ? 'text-white' : 'text-slate-900'}`}>{f.title}</h4><p className={`text-sm ${isDark ? 'text-slate-400' : 'text-slate-500'}`}>{f.desc}</p></div>))}
            </div>
          </div>
        )}

        {/* PARAPHRASER VIEW */}
        {currentView === 'paraphrase' && (
          <div className="w-full max-w-5xl animate-fade-in mt-8">
            <div className="flex justify-between items-end mb-8"><h2 className={`text-4xl font-black tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>Smart Paraphraser</h2></div>
            <div className="grid md:grid-cols-2 gap-6">
              <div className={`p-6 rounded-3xl border flex flex-col h-[500px] ${isDark ? 'bg-slate-900/50 border-white/10' : 'bg-white border-slate-200'}`}>
                <h3 className="text-sm font-bold uppercase text-slate-500 mb-4 flex gap-2"><FileText size={16}/> Input</h3>
                <textarea className={`flex-grow bg-transparent resize-none focus:outline-none text-lg p-2 rounded-lg ${isDark ? 'text-slate-200 placeholder-slate-600' : 'text-slate-800 placeholder-slate-400'}`} placeholder="Paste text to rewrite..." value={paraInput} onChange={(e) => setParaInput(e.target.value)} />
                <div className="mt-4 flex gap-2 overflow-x-auto pb-2 scrollbar-hide">
                   {['Standard', 'Fluent', 'Formal'].map(t => (<button key={t} onClick={() => setTone(t)} className={`px-4 py-2 rounded-full text-xs font-bold border transition-all ${tone === t ? 'bg-indigo-600 text-white border-indigo-600' : isDark ? 'bg-slate-800 text-slate-400 border-slate-700' : 'bg-slate-100 text-slate-500 border-slate-200'}`}>{t}</button>))}
                </div>
              </div>
              <div className={`p-6 rounded-3xl border flex flex-col h-[500px] relative ${isDark ? 'bg-indigo-950/20 border-indigo-500/20' : 'bg-indigo-50 border-indigo-200'}`}>
                <h3 className="text-sm font-bold uppercase text-indigo-400 mb-4 flex gap-2"><Sparkles size={16}/> Editor (Click words or sentences)</h3>
                {paraLoading ? (
                  <div className="flex-grow flex flex-col items-center justify-center text-indigo-400 gap-4"><Loader2 className="animate-spin" size={32}/><span>Rewriting as {tone}...</span></div>
                ) : (
                  <div className={`flex-grow overflow-y-auto p-4 leading-relaxed text-lg ${isDark ? 'text-indigo-100' : 'text-indigo-900'}`}>
                    {paraSentences.length > 0 ? (
                      <div className="text-left">
                        {paraSentences.map((sent, sIdx) => (
                          <span key={sIdx} className="group relative mr-1 hover:bg-indigo-500/10 rounded px-1 transition-colors">
                            {sent.split(' ').map((word, wIdx) => (
                              <span key={wIdx} className="relative inline-block mr-1">
                                 <span onClick={(e) => handleWordClick(e, word, sIdx, wIdx)} className={`cursor-pointer rounded hover:text-indigo-400 hover:underline decoration-dashed underline-offset-4 ${activeWord.idx === wIdx && activeWord.sentIdx === sIdx ? 'bg-indigo-500 text-white px-1' : ''}`}>{word}</span>
                              </span>
                            ))}
                            <span onClick={(e) => handleSentenceClick(e, sent, sIdx)} className="inline-flex items-center justify-center w-5 h-5 ml-0.5 rounded-full bg-indigo-500/20 text-indigo-400 cursor-pointer hover:bg-indigo-500 hover:text-white transition-all opacity-0 group-hover:opacity-100 align-middle transform translate-y-[-2px]" title="Rewrite this sentence"><Repeat size={10} /></span>
                          </span>
                        ))}
                      </div>
                    ) : <span className="opacity-40 italic">Result appears here...</span>}
                  </div>
                )}
                {paraSentences.length > 0 && <button onClick={() => navigator.clipboard.writeText(paraSentences.join(' '))} className="absolute bottom-6 right-6 p-3 rounded-xl bg-indigo-500 text-white shadow-lg"><Copy size={20}/></button>}
              </div>
            </div>
            <div className="flex justify-center mt-8"><button onClick={handleParaphrase} disabled={paraLoading || !paraInput} className="bg-indigo-600 hover:bg-indigo-500 text-white px-10 py-4 rounded-xl font-bold shadow-lg flex gap-3">{paraLoading ? "Processing..." : "Paraphrase Now"} <RefreshCw size={20}/></button></div>
          </div>
        )}
        
        {/* HISTORY VIEW */}
        {currentView === 'history' && (
           <div className="w-full max-w-5xl animate-fade-in mt-8">
             <div className="flex justify-between items-end mb-8"><h2 className={`text-4xl font-black tracking-tight ${isDark ? 'text-white' : 'text-slate-900'}`}>Activity Log</h2> {history.length > 0 && <button onClick={clearHistory} className="flex items-center gap-2 px-4 py-2 rounded-lg bg-red-500/10 text-red-500 font-bold text-sm border border-red-500/20"><Trash2 size={16} /> Clear</button>}</div>
             {history.length > 0 ? (
               <div className="grid gap-4">{history.map((item) => (<div key={item.id} onClick={() => restoreHistoryItem(item)} className={`group relative p-6 rounded-2xl border transition-all cursor-pointer hover:scale-[1.01] ${isDark ? 'bg-slate-900/50 border-white/5 hover:border-blue-500/50' : 'bg-white border-slate-200 hover:border-blue-500/50'}`}><div className="flex justify-between items-center"><div className="flex-grow pr-8"><div className="flex items-center gap-3 mb-2">{item.type === 'scan' ? <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${item.risk === 'High' ? 'bg-red-500/20 text-red-500' : item.risk === 'Medium' ? 'bg-yellow-500/20 text-yellow-600' : 'bg-emerald-500/20 text-emerald-500'}`}>{item.prediction}</span> : <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-indigo-500/20 text-indigo-400">{item.tone} Paraphrase</span>}<span className="text-slate-400 text-xs font-mono">{item.date} • {item.time}</span></div><p className={`font-medium line-clamp-1 ${isDark ? 'text-slate-300' : 'text-slate-700'}`}>"{item.type === 'scan' ? item.textSnippet : item.original}"</p></div>{item.type === 'scan' ? <div className={`text-xl font-black ${item.risk === 'High' ? 'text-red-500' : item.risk === 'Medium' ? 'text-yellow-500' : 'text-emerald-500'}`}>{item.confidence}%</div> : <div className="text-indigo-500"><RefreshCw size={24}/></div>}</div></div>))}</div>
             ) : (
               <div className="p-12 text-center rounded-3xl border border-dashed border-slate-300 bg-slate-50/50">
                 <div className="inline-flex p-4 rounded-full bg-slate-200/50 mb-4 text-slate-400"><History size={32} /></div>
                 <h3 className="text-xl font-bold text-slate-900">No history found</h3>
                 <p className="text-slate-500 mt-2">{currentUser ? "Start scanning or paraphrasing to see logs here." : "Please login to save and view your history."}</p>
               </div>
             )}
           </div>
        )}
      </main>
    </div>
  );
}

export default App;