import React, { useState, useEffect } from 'react';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Settings, 
  Play, 
  AlertCircle, 
  BarChart2, 
  Layers, 
  Clock, 
  Search,
  ChevronRight,
  ShieldAlert,
  Zap
} from 'lucide-react';

// --- MOCK DATA & SIMULATION LOGIC ---

const ASSETS = [
  { symbol: 'AAPL', name: 'Apple Inc.', type: 'Stock', price: 173.50 },
  { symbol: 'BTC', name: 'Bitcoin', type: 'Crypto', price: 64230.00 },
  { symbol: 'EUR/USD', name: 'Euro / US Dollar', type: 'Forex', price: 1.0845 },
  { symbol: 'NVDA', name: 'NVIDIA Corp.', type: 'Stock', price: 884.20 },
];

const TIMEFRAMES = ['15M', '1H', '4H', '1D', '1W'];

// Simulates running the ensemble model
const generatePrediction = (asset) => {
  const isBullish = Math.random() > 0.4; // Slight bullish bias for the simulation
  const mainConfidence = Math.floor(Math.random() * 40) + 50; // 50-90%
  
  const models = [
    { name: 'Deep Neural Net (LSTM)', weight: 35, signal: isBullish ? 'Buy' : 'Sell', conf: mainConfidence + (Math.random() * 10 - 5) },
    { name: 'Gradient Boosting (XGB)', weight: 25, signal: isBullish ? 'Buy' : (Math.random() > 0.5 ? 'Sell' : 'Hold'), conf: mainConfidence - 10 + (Math.random() * 15) },
    { name: 'Mean Reversion', weight: 20, signal: Math.random() > 0.5 ? 'Hold' : (isBullish ? 'Buy' : 'Sell'), conf: 45 + Math.random() * 30 },
    { name: 'NLP Sentiment Analysis', weight: 20, signal: isBullish ? 'Buy' : 'Sell', conf: 60 + Math.random() * 30 },
  ];

  // Calculate overall weighted signal
  let score = 0;
  models.forEach(m => {
    let val = m.signal === 'Buy' ? 1 : m.signal === 'Sell' ? -1 : 0;
    score += val * (m.weight / 100) * (m.conf / 100);
  });

  let finalSignal = 'Hold';
  if (score > 0.2) finalSignal = 'Buy';
  if (score < -0.2) finalSignal = 'Sell';
  if (score > 0.6) finalSignal = 'Strong Buy';
  if (score < -0.6) finalSignal = 'Strong Sell';

  const volatility = Math.random() * 3 + 1; // 1-4%
  const currentPrice = asset.price;
  const targetDiff = currentPrice * (volatility / 100) * (finalSignal.includes('Buy') ? 1 : -1);
  const targetPrice = currentPrice + targetDiff;
  const stopLoss = currentPrice - (targetDiff * 0.5);

  // Generate mock chart data (historical + projection)
  const chartData = [];
  let p = currentPrice - (currentPrice * (Math.random() * 0.05)); // Start a bit lower/higher
  for(let i=0; i<20; i++) {
    chartData.push(p);
    p += (Math.random() - 0.5) * (currentPrice * 0.01);
  }
  chartData.push(currentPrice);
  
  const projectionData = [currentPrice];
  let projP = currentPrice;
  const step = targetDiff / 5;
  for(let i=0; i<5; i++) {
    projP += step + ((Math.random() - 0.5) * step * 0.5);
    projectionData.push(projP);
  }

  return {
    signal: finalSignal,
    confidence: Math.min(99, Math.abs(score * 100).toFixed(1)),
    targetPrice: targetPrice.toFixed(asset.type === 'Forex' ? 4 : 2),
    stopLoss: stopLoss.toFixed(asset.type === 'Forex' ? 4 : 2),
    models: models,
    chartData,
    projectionData
  };
};

// --- COMPONENTS ---

// Simple SVG Line Chart Component
const MiniChart = ({ data, projection, colorClass }) => {
  if (!data || data.length === 0) return null;
  
  const allData = [...data, ...projection];
  const min = Math.min(...allData);
  const max = Math.max(...allData);
  const range = max - min || 1;
  
  const width = 300;
  const height = 100;
  
  const formatPoints = (arr, startX, stepX) => {
    return arr.map((val, i) => {
      const x = startX + (i * stepX);
      const y = height - ((val - min) / range) * height;
      return `${x},${y}`;
    }).join(' ');
  };

  const stepX = width / (allData.length - 1);
  const histPoints = formatPoints(data, 0, stepX);
  const projStartX = (data.length - 1) * stepX;
  const projPoints = formatPoints(projection, projStartX, stepX);

  return (
    <div className="w-full h-32 relative flex items-center justify-center">
      <svg viewBox={`0 -10 ${width} ${height + 20}`} className="w-full h-full overflow-visible">
        {/* Historical Data */}
        <polyline 
          points={histPoints} 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="3" 
          className="text-slate-400 opacity-50"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Projection Data */}
        <polyline 
          points={projPoints} 
          fill="none" 
          stroke="currentColor" 
          strokeWidth="3" 
          strokeDasharray="6,6"
          className={colorClass}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {/* Current Price Node */}
        <circle 
          cx={projStartX} 
          cy={height - ((data[data.length-1] - min) / range) * height} 
          r="4" 
          className="fill-current text-white" 
        />
        {/* Target Price Node */}
        <circle 
          cx={width} 
          cy={height - ((projection[projection.length-1] - min) / range) * height} 
          r="4" 
          className={`fill-current ${colorClass}`} 
        />
      </svg>
    </div>
  );
};


export default function App() {
  const [selectedAsset, setSelectedAsset] = useState(ASSETS[0]);
  const [timeframe, setTimeframe] = useState('1D');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');

  // Initial load simulation
  useEffect(() => {
    runAnalysis(ASSETS[0]);
  }, []);

  const runAnalysis = (asset = selectedAsset) => {
    setIsAnalyzing(true);
    // Simulate network delay and calculation time
    setTimeout(() => {
      const prediction = generatePrediction(asset);
      setResults(prediction);
      setIsAnalyzing(false);
    }, 1200);
  };

  const handleAssetSelect = (asset) => {
    setSelectedAsset(asset);
    setResults(null);
    runAnalysis(asset);
  };

  const filteredAssets = ASSETS.filter(a => 
    a.symbol.toLowerCase().includes(searchQuery.toLowerCase()) || 
    a.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getSignalColor = (signal) => {
    if (!signal) return 'text-slate-400';
    if (signal.includes('Buy')) return 'text-emerald-400';
    if (signal.includes('Sell')) return 'text-rose-400';
    return 'text-amber-400';
  };

  const getSignalBg = (signal) => {
    if (!signal) return 'bg-slate-800';
    if (signal.includes('Buy')) return 'bg-emerald-400/10 border-emerald-500/30';
    if (signal.includes('Sell')) return 'bg-rose-400/10 border-rose-500/30';
    return 'bg-amber-400/10 border-amber-500/30';
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 font-sans selection:bg-indigo-500/30 flex">
      
      {/* SIDEBAR */}
      <aside className="w-20 lg:w-64 bg-slate-900 border-r border-slate-800 flex flex-col transition-all duration-300">
        <div className="h-16 flex items-center justify-center lg:justify-start lg:px-6 border-b border-slate-800">
          <Zap className="w-8 h-8 text-indigo-500" />
          <span className="ml-3 font-bold text-lg hidden lg:block tracking-tight text-white">NexTrade<span className="text-indigo-500">AI</span></span>
        </div>
        
        <nav className="flex-1 py-6 flex flex-col gap-2 px-3">
          <button className="flex items-center gap-3 px-3 py-3 rounded-xl bg-indigo-500/10 text-indigo-400 transition-colors">
            <Activity className="w-5 h-5" />
            <span className="font-medium hidden lg:block">Live Models</span>
          </button>
          <button className="flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
            <BarChart2 className="w-5 h-5" />
            <span className="font-medium hidden lg:block">Backtesting</span>
          </button>
          <button className="flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors">
            <Layers className="w-5 h-5" />
            <span className="font-medium hidden lg:block">Ensemble Config</span>
          </button>
        </nav>

        <div className="p-4 border-t border-slate-800">
          <button className="flex items-center gap-3 px-3 py-3 rounded-xl hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors w-full">
            <Settings className="w-5 h-5" />
            <span className="font-medium hidden lg:block">Settings</span>
          </button>
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col h-screen overflow-hidden">
        
        {/* HEADER */}
        <header className="h-16 bg-slate-900/50 backdrop-blur-sm border-b border-slate-800 flex items-center justify-between px-6 z-10">
          <div className="flex items-center bg-slate-800/50 rounded-lg px-3 py-1.5 border border-slate-700/50 w-64 focus-within:border-indigo-500/50 transition-colors">
            <Search className="w-4 h-4 text-slate-400" />
            <input 
              type="text" 
              placeholder="Search assets..." 
              className="bg-transparent border-none outline-none text-sm ml-2 w-full text-slate-200 placeholder-slate-500"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
            />
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-slate-400 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700/50">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
              API Connected
            </div>
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-indigo-500 to-purple-500 flex items-center justify-center text-sm font-bold text-white shadow-lg">
              JS
            </div>
          </div>
        </header>

        {/* DASHBOARD SCROLL AREA */}
        <div className="flex-1 overflow-auto p-6 lg:p-8 custom-scrollbar">
          
          <div className="max-w-6xl mx-auto space-y-6">
            
            {/* TOP CONTROLS & ASSET INFO */}
            <div className="flex flex-col md:flex-row justify-between items-start md:items-end gap-4">
              <div>
                <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                  {selectedAsset.symbol} 
                  <span className="text-lg font-normal text-slate-400 bg-slate-800 px-2 py-0.5 rounded-md border border-slate-700">{selectedAsset.type}</span>
                </h1>
                <p className="text-slate-400 mt-1">{selectedAsset.name}</p>
              </div>

              <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto pb-2 md:pb-0">
                {/* Timeframe Selector */}
                <div className="flex bg-slate-800/80 rounded-lg p-1 border border-slate-700/50">
                  {TIMEFRAMES.map(tf => (
                    <button
                      key={tf}
                      onClick={() => { setTimeframe(tf); runAnalysis(selectedAsset); }}
                      className={`px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${timeframe === tf ? 'bg-indigo-500 text-white shadow-md' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/50'}`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>

                <button 
                  onClick={() => runAnalysis()}
                  disabled={isAnalyzing}
                  className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-500 text-white px-5 py-2 rounded-lg font-medium transition-all shadow-lg shadow-indigo-500/20 disabled:opacity-70 disabled:cursor-not-allowed"
                >
                  {isAnalyzing ? (
                    <Activity className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 fill-current" />
                  )}
                  {isAnalyzing ? 'Analyzing...' : 'Run Ensemble'}
                </button>
              </div>
            </div>

            {/* ASSET SELECTOR CAROUSEL */}
            <div className="flex gap-3 overflow-x-auto pb-2 custom-scrollbar">
              {filteredAssets.map(asset => (
                <button
                  key={asset.symbol}
                  onClick={() => handleAssetSelect(asset)}
                  className={`flex-shrink-0 flex items-center justify-between p-3 rounded-xl border transition-all min-w-[200px] text-left
                    ${selectedAsset.symbol === asset.symbol 
                      ? 'bg-slate-800 border-indigo-500 shadow-[0_0_15px_rgba(99,102,241,0.15)]' 
                      : 'bg-slate-900 border-slate-800 hover:border-slate-700'
                    }`}
                >
                  <div>
                    <div className="font-bold text-white">{asset.symbol}</div>
                    <div className="text-xs text-slate-400 truncate w-24">{asset.name}</div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm text-slate-200">
                      ${asset.price.toLocaleString(undefined, {minimumFractionDigits: 2})}
                    </div>
                  </div>
                </button>
              ))}
            </div>

            {/* RESULTS GRID */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 pt-4">
              
              {/* MAIN SIGNAL CARD */}
              <div className={`col-span-1 lg:col-span-1 rounded-2xl border p-6 flex flex-col justify-between transition-all duration-500 relative overflow-hidden
                ${isAnalyzing ? 'bg-slate-900 border-slate-800' : getSignalBg(results?.signal)}
              `}>
                {/* Background Glow */}
                {!isAnalyzing && results && (
                  <div className={`absolute -top-20 -right-20 w-40 h-40 blur-3xl opacity-20 rounded-full
                    ${results.signal.includes('Buy') ? 'bg-emerald-500' : results.signal.includes('Sell') ? 'bg-rose-500' : 'bg-amber-500'}
                  `} />
                )}

                <div>
                  <div className="flex items-center justify-between mb-2">
                    <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase">Ensemble Signal</h2>
                    <Clock className="w-4 h-4 text-slate-500" />
                  </div>
                  
                  {isAnalyzing ? (
                    <div className="py-8 flex flex-col items-center justify-center space-y-4">
                      <div className="w-12 h-12 border-4 border-indigo-500/30 border-t-indigo-500 rounded-full animate-spin"></div>
                      <p className="text-indigo-400 animate-pulse font-medium text-sm">Aggregating Models...</p>
                    </div>
                  ) : results ? (
                    <>
                      <div className="mt-4 flex items-end gap-3">
                        <span className={`text-5xl font-extrabold tracking-tight ${getSignalColor(results.signal)}`}>
                          {results.signal}
                        </span>
                        {results.signal.includes('Buy') && <TrendingUp className={`w-8 h-8 mb-1 ${getSignalColor(results.signal)}`} />}
                        {results.signal.includes('Sell') && <TrendingDown className={`w-8 h-8 mb-1 ${getSignalColor(results.signal)}`} />}
                      </div>
                      
                      <div className="mt-6 space-y-4 bg-slate-950/40 p-4 rounded-xl backdrop-blur-sm border border-white/5">
                        <div className="flex justify-between items-center">
                          <span className="text-slate-400 text-sm">Model Confidence</span>
                          <span className="font-mono text-white font-bold">{results.confidence}%</span>
                        </div>
                        {/* Progress bar */}
                        <div className="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${results.signal.includes('Buy') ? 'bg-emerald-500' : results.signal.includes('Sell') ? 'bg-rose-500' : 'bg-amber-500'}`} 
                            style={{ width: `${results.confidence}%` }}
                          />
                        </div>
                      </div>
                    </>
                  ) : null}
                </div>

                {!isAnalyzing && results && (
                   <div className="mt-6 grid grid-cols-2 gap-3 pt-4 border-t border-white/10">
                   <div>
                     <p className="text-xs text-slate-400 mb-1">Target Price</p>
                     <p className="font-mono font-bold text-slate-200">${results.targetPrice}</p>
                   </div>
                   <div>
                     <p className="text-xs text-slate-400 mb-1">Stop Loss</p>
                     <p className="font-mono font-bold text-slate-200">${results.stopLoss}</p>
                   </div>
                 </div>
                )}
              </div>

              {/* CHART / VISUALIZATION */}
              <div className="col-span-1 lg:col-span-2 bg-slate-900 border border-slate-800 rounded-2xl p-6 flex flex-col relative overflow-hidden">
                <div className="flex justify-between items-center mb-6">
                  <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase flex items-center gap-2">
                    <Activity className="w-4 h-4" /> Trajectory Projection
                  </h2>
                  <div className="flex gap-2 text-xs">
                    <span className="flex items-center gap-1 text-slate-400"><div className="w-2 h-0.5 bg-slate-400"></div> Historical</span>
                    <span className="flex items-center gap-1 text-slate-400"><div className="w-2 h-0.5 bg-indigo-400 border border-dashed"></div> Predicted</span>
                  </div>
                </div>

                {isAnalyzing ? (
                  <div className="flex-1 flex items-center justify-center min-h-[200px]">
                    <Activity className="w-8 h-8 text-slate-700 animate-pulse" />
                  </div>
                ) : results ? (
                  <div className="flex-1 flex flex-col justify-center min-h-[200px]">
                    <MiniChart 
                      data={results.chartData} 
                      projection={results.projectionData} 
                      colorClass={results.signal.includes('Buy') ? 'text-emerald-500' : results.signal.includes('Sell') ? 'text-rose-500' : 'text-amber-500'}
                    />
                    <div className="flex justify-between mt-4 text-xs text-slate-500 font-mono">
                      <span>{timeframe === '1D' ? '30 Days Ago' : 'Past Period'}</span>
                      <span>Current</span>
                      <span>Projected Target</span>
                    </div>
                  </div>
                ) : null}
              </div>

              {/* ENSEMBLE BREAKDOWN */}
              <div className="col-span-1 lg:col-span-3 bg-slate-900 border border-slate-800 rounded-2xl p-6">
                 <div className="flex justify-between items-center mb-6">
                  <h2 className="text-sm font-semibold tracking-wider text-slate-400 uppercase flex items-center gap-2">
                    <Layers className="w-4 h-4" /> Model Composition & Weights
                  </h2>
                  <button className="text-indigo-400 hover:text-indigo-300 text-sm flex items-center gap-1 transition-colors">
                    Adjust Weights <ChevronRight className="w-4 h-4" />
                  </button>
                </div>

                {isAnalyzing ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {[1,2,3,4].map(i => (
                      <div key={i} className="bg-slate-800/50 rounded-xl p-4 h-24 animate-pulse border border-slate-700/50"></div>
                    ))}
                  </div>
                ) : results ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    {results.models.map((model, idx) => (
                      <div key={idx} className="bg-slate-950/50 border border-slate-800 rounded-xl p-4 hover:border-slate-700 transition-colors">
                        <div className="flex justify-between items-start mb-3">
                          <h3 className="font-medium text-slate-200 text-sm w-3/4 leading-tight">{model.name}</h3>
                          <span className="text-xs font-mono bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded border border-slate-700">
                            {model.weight}%
                          </span>
                        </div>
                        
                        <div className="flex items-end justify-between mt-auto pt-2">
                          <div className={`font-bold text-sm ${getSignalColor(model.signal)}`}>
                            {model.signal}
                          </div>
                          <div className="text-right">
                            <span className="text-xs text-slate-500 block mb-1">Confidence</span>
                            <div className="font-mono text-sm text-slate-300">{model.conf.toFixed(1)}%</div>
                          </div>
                        </div>

                         <div className="w-full bg-slate-800 rounded-full h-1 mt-3 overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${model.signal.includes('Buy') ? 'bg-emerald-500' : model.signal.includes('Sell') ? 'bg-rose-500' : 'bg-amber-500'}`} 
                            style={{ width: `${model.conf}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                ) : null}
              </div>

              {/* RISK WARNING */}
              <div className="col-span-1 lg:col-span-3 flex items-start gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-500/80 text-sm">
                <ShieldAlert className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <p>
                  <strong>Disclaimer:</strong> This model ensemble is for informational purposes only and does not constitute financial advice. AI-driven predictions are based on historical data patterns and sentiment analysis, which cannot guarantee future performance. Always manage your risk appropriately.
                </p>
              </div>

            </div>
          </div>
        </div>
      </main>

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
          height: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: transparent;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #334155;
          border-radius: 10px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #475569;
        }
      `}} />
    </div>
  );
}
