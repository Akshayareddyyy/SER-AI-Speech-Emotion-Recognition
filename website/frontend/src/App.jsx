import React, { useState, useRef, useEffect } from 'react'
import './App.css'

const API_URL = "http://127.0.0.1:8000/predict"

const EMOJI = {
  happy: '🙂',
  sad: '☂️',
  angry: '⚡',
  fearful: '😧',
  disgust: '🤢',
  neutral: '•',
  calm: '▫️',
  surprised: '✨'
}

const ALLOWED_EMOTIONS = ['calm','happy','sad','angry','fearful','neutral','disgust','surprised']

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [audioURL, setAudioURL] = useState(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState('Ready')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [level, setLevel] = useState(0)
  const fileRef = useRef()
  const gameFileRef = useRef()

  // Quest state
  const [targetEmotion, setTargetEmotion] = useState(null)
  const [promptText, setPromptText] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const mediaRecorderRef = useRef(null)
  const recordedChunksRef = useRef([])
  const streamRef = useRef(null)

  const [gameScore, setGameScore] = useState(0)
  const [streak, setStreak] = useState(0)
  const [gameFile, setGameFile] = useState(null)
  const [gameAudioURL, setGameAudioURL] = useState(null)
  const [gameResult, setGameResult] = useState(null)
  const [gameLoading, setGameLoading] = useState(false)

  useEffect(() => { pickNewTarget() }, [])

  function formatConfidence(value) {
    if (value === undefined || value === null) return '0.00'
    return value <= 1 ? (value * 100).toFixed(2) : value.toFixed(2)
  }

  function getEmotionEmoji(name) {
    return EMOJI[name] || '•'
  }

  function reset() {
    setSelectedFile(null)
    setAudioURL(null)
    setResult(null)
    setError('')
    setStatus('Ready')
    if (fileRef.current) fileRef.current.value = null
  }

  function resetGame() {
    setGameFile(null)
    setGameAudioURL(null)
    setGameResult(null)
    setGameLoading(false)
    if (gameFileRef.current) gameFileRef.current.value = null
  }

  function handleFile(e) {
    const f = e.target.files[0]
    setResult(null)
    setError('')
    if (!f) return
    setSelectedFile(f)
    setAudioURL(URL.createObjectURL(f))
    // auto-start prediction
    handlePredict(f)
  }

  function handleGameFile(e) {
    const f = e.target.files[0]
    setGameResult(null)
    if (!f) return
    setGameFile(f)
    setGameAudioURL(URL.createObjectURL(f))
  }

  async function handlePredict(fileArg = null) {
    const fileToSend = fileArg || selectedFile
    if (!fileToSend) {
      setError('Please select an audio file')
      return
    }

    setLoading(true)
    setStatus('Analyzing...')
    setError('')

    try {
      const formData = new FormData()
      formData.append('file', fileToSend)

      const response = await fetch(API_URL, { method: 'POST', body: formData })
      let data
      try {
        data = await response.json()
      } catch (parseErr) {
        const text = await response.text().catch(() => '')
        data = { error: text || String(parseErr) }
      }
      if (!response.ok) throw new Error(data.error || 'Prediction failed')

      setResult({
        emotion: data.emotion,
        confidence: data.confidence,
        model_version: data.model_version,
        top_predictions: data.top_predictions || []
      })
      setStatus('Done')
    } catch (err) {
      console.error(err)
      setError(String(err))
      setStatus('Error')
    } finally {
      setLoading(false)
    }
  }

  async function handleGamePredict() {
    if (!gameFile) return setError('Please select or record audio for the challenge')
    if (!targetEmotion) return setError('No target emotion selected')

    setGameLoading(true)
    setError('')
    try {
      const formData = new FormData()
      // If this is a recorded blob file (created by our recorder), ensure filename is 'recorded_audio.webm'
      if (gameFile && typeof gameFile.name === 'string' && gameFile.name.toLowerCase().includes('record')) {
        formData.append('file', gameFile, 'recorded_audio.webm')
      } else {
        formData.append('file', gameFile)
      }

      const response = await fetch(API_URL, { method: 'POST', body: formData })
      let data
      try {
        data = await response.json()
      } catch (parseErr) {
        const text = await response.text().catch(() => '')
        data = { error: text || String(parseErr) }
      }
      if (!response.ok) throw new Error(data.error || 'Prediction failed')

      const predicted = data.emotion
      const confidence = data.confidence
      const top_predictions = data.top_predictions || []

      const correct = predicted === targetEmotion

      let newStreak = correct ? (streak + 1) : 0
      let delta = 0
      if (correct) {
        delta += 10
        if (confidence >= 0.7 || confidence >= 70) delta += 5
        // streak bonus: +2 for each consecutive correct round beyond the first
        delta += 2 * Math.max(0, newStreak - 1)
      } else {
        newStreak = 0
      }

      setStreak(newStreak)
      setGameScore(s => s + delta)
      setGameResult({ predicted, confidence, top_predictions, correct })
    } catch (err) {
      console.error(err)
      setError(String(err))
    } finally {
      setGameLoading(false)
    }
  }

  function pickNewTarget() {
    const prompts = {
      happy: 'Today is going to be amazing!',
      sad: 'I really miss those days.',
      angry: 'This is not fair at all.',
      calm: 'Everything is going to be okay.',
      fearful: 'I think someone is behind me.',
      surprised: "I can't believe this happened!",
      disgust: 'This smells really bad.',
      neutral: 'I will meet you tomorrow.'
    }
    const choice = ALLOWED_EMOTIONS[Math.floor(Math.random() * ALLOWED_EMOTIONS.length)]
    setTargetEmotion(choice)
    setPromptText(prompts[choice] || '')
    // reset round result but keep score/streak
    setGameResult(null)
    setGameFile(null)
    setGameAudioURL(null)
    if (gameFileRef.current) gameFileRef.current.value = null
  }

  // Recording helpers
  async function startRecording() {
    setError('')
    console.log('startRecording clicked')
    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setError('Browser does not support microphone access')
        return
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      // pick a supported mimeType when possible
      let options = {}
      try {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported) {
          if (MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) options.mimeType = 'audio/webm;codecs=opus'
          else if (MediaRecorder.isTypeSupported('audio/webm')) options.mimeType = 'audio/webm'
          else if (MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')) options.mimeType = 'audio/ogg;codecs=opus'
        }
      } catch (e) {
        // ignore and let constructor choose default
      }
      mediaRecorderRef.current = options.mimeType ? new MediaRecorder(stream, options) : new MediaRecorder(stream)
      recordedChunksRef.current = []
      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data)
      }
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(recordedChunksRef.current, { type: recordedChunksRef.current[0]?.type || 'audio/webm' })
        const file = new File([blob], 'recorded_audio.webm', { type: 'audio/webm' })
        setGameFile(file)
        setGameAudioURL(URL.createObjectURL(blob))
        // stop tracks
        try {
          if (streamRef.current) {
            streamRef.current.getTracks().forEach(t => t.stop())
            streamRef.current = null
          }
        } catch (err) {}
      }
      try{
        mediaRecorderRef.current.start()
        setIsRecording(true)
      } catch(startErr){
        console.error('MediaRecorder start failed', startErr)
        setError('Recording failed to start')
      }
    } catch (err) {
      console.error('startRecording error', err)
      setError('Microphone access denied or not available')
    }
  }

  function stopRecording() {
    console.log('stopRecording clicked')
    try {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      // also stop tracks if any
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop())
        streamRef.current = null
      }
    } catch (err) {
      console.error('stopRecording error', err)
      setError('Error stopping recording')
    }
    setIsRecording(false)
  }

  // Drag and drop
  function handleDrop(e) {
    e.preventDefault()
    const f = e.dataTransfer.files && e.dataTransfer.files[0]
    if (f) {
      setSelectedFile(f)
      setAudioURL(URL.createObjectURL(f))
      handlePredict(f)
    }
  }

  function handleDragOver(e) { e.preventDefault() }

  return (
    <div className="app-shell">
      <nav className="topnav">
          <div className="container nav-inner">
          <div className="nav-left">
            <div className="brand">
              <span className="brand-icon" aria-hidden="true">
                <span className="bar b1" />
                <span className="bar b2" />
                <span className="bar b3" />
              </span>
              <span className="brand-text">SER AI</span>
            </div>
          </div>
          <div className="nav-right">
            <a href="#home">Home</a>
            <a href="#detector">Detector</a>
            <a href="#quest">Quest</a>
            <a href="#about">About</a>
          </div>
        </div>
      </nav>

      <main>
        <header id="home" className="hero-outer full-hero">
          <div className="hero-bg-decor" />

          <div className="hero-inner container">
            <div className="hero-grid">
              <div className="hero-left compact">
                <h1 className="hero-title">SER AI</h1>
                <div className="hero-sub">Speech Emotion Recognition</div>
                <p className="desc short">Detect emotional tone from speech audio using AI-powered voice intelligence.</p>

                <div className="hero-ctas">
                  <button className="cta" onClick={() => document.getElementById('file-input') && document.getElementById('file-input').click()}>Try Detector</button>
                </div>

                <div className="hero-chips compact">
                  <span>Voice Intelligence</span>
                  <span>8 Emotions</span>
                </div>
              </div>

              <div className="hero-right">
                <div className="hero-visual-3d large clean">
                  <div className="voice-orb" aria-hidden="true">
                    <div className="glass-sheen" />
                    <div className="glow-orb" />
                    <div className="orbit orbit-one" />
                    <div className="orbit orbit-two" />

                    <div className="mic-3d">
                      <div className="mic-glass" />
                      <div className="mic-capsule" />
                      <div className="mic-pole" />
                    </div>
                    <span className="emotion-chip chip-calm">calm</span>
                    <span className="emotion-chip chip-angry">angry</span>
                  </div>

                  <div className="wave-equalizer" aria-hidden="true">
                    <span></span><span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span><span></span>
                  </div>

                  <div className="mini-preview">
                    <div className="mini-title">Detected Emotion</div>
                    <div className="mini-emotion">calm</div>
                    <div className="mini-confidence">92%</div>
                    <div className="mini-bars">
                      <div className="mini-bar" style={{width:'78%'}}></div>
                      <div className="mini-bar" style={{width:'12%'}}></div>
                      <div className="mini-bar" style={{width:'10%'}}></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </header>

        <section id="detector" className="detector">
          <div className="container">
            <div className="detector-shell">
            <div className="detector-header">
              <h2>Detector</h2>
              <p className="muted">Upload an audio file to analyze emotional tone.</p>
            </div>

              <div className="detector-card card">
                <div className="detector-grid">
                <div className="upload-panel detector-section upload-section" onDrop={handleDrop} onDragOver={handleDragOver}>
                  <input id="file-input" ref={fileRef} type="file" accept="audio/*" onChange={handleFile} />
                  <div className="upload-drop">
                    <div className="upload-text">Drag & drop audio here or click to select</div>
                    <div className="accepted muted">Accepted: .wav .mp3 .m4a .flac</div>
                    <div className="upload-actions">
                      <button className="select" onClick={() => fileRef.current && fileRef.current.click()}>Choose File</button>
                      <button className={`predict ${loading ? 'working' : ''}`} onClick={() => handlePredict()} disabled={loading || !selectedFile}>{loading ? 'Analyzing…' : 'Analyze'}</button>
                      <button className="reset" onClick={reset}>Reset</button>
                    </div>
                  </div>
                </div>

                <div className="player-panel detector-section preview-section">
                  <div className="player card">
                        {audioURL ? (
                      <>
                        <audio id="audio-player" controls src={audioURL} style={{ width: '100%' }} />
                        <div className="file-name muted">{selectedFile && selectedFile.name}</div>
                        <div style={{marginTop:12}}>
                          {/* audio preview controls (visualizer moved to hero) */}
                        </div>
                      </>
                    ) : (
                      <div className="preview-placeholder muted">
                        <div className="ph-title">No audio selected</div>
                        <div className="ph-sub">Select or drop an audio file to preview.</div>
                      </div>
                    )}
                  </div>
                </div>

                <div className="prediction-panel detector-section prediction-section">
                  <div className="prediction-card card">
                    <div className="prediction-top">
                      <div>
                        <div className="pred-label">Predicted Emotion</div>
                        <div className="prediction-main">
                          <div className="emotion-name">{result ? result.emotion : '—'}</div>
                          <div className="confidence muted">Confidence: {result ? formatConfidence(result.confidence) + '%' : '—'}</div>
                          <div className="model-version muted">Model: {result ? result.model_version : '—'}</div>
                        </div>
                      </div>
                    </div>

                    <div className="top-bars">
                      {(result && result.top_predictions ? result.top_predictions.slice(0,3) : []).map((p, i) => (
                        <div className="top-bar" key={i}>
                          <div className="bar-meta"><div className="name">{p.emotion}</div><div className="percent">{formatConfidence(p.confidence)}%</div></div>
                          <div className="bar-track"><div className="bar-fill" style={{ width: `${Math.min(100, (p.confidence<=1?p.confidence*100:p.confidence))}%` }} /></div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            </div>
          </div>
        </section>

        <section id="quest" className="quest-section">
          <div className="container">
            <div className="quest-shell">
              <div className="quest-header">
                <h2>Emotion Quest</h2>
                <p className="muted">A voice challenge: act the target emotion and see if the AI recognizes it.</p>
              </div>

              <div className="quest-grid">
                <div className="quest-left">
                  <div className="mission-card">
                    <div className="mission-title">Mission</div>
                    <div className="mission-emotion">{targetEmotion || '—'}</div>
                    <div className="mission-prompt">{promptText || 'Press "Next Challenge" to begin.'}</div>
                  </div>

                  <div className="quest-actions">
                    <div className="muted">Record or upload a short clip (1-5s)</div>
                    <div style={{marginTop:8}}>
                      <button className="select" onClick={() => gameFileRef.current && gameFileRef.current.click()}>{gameFile ? 'Change File' : 'Choose File'}</button>
                      <button className={`select ${isRecording? 'recording':''}`} onClick={() => isRecording ? stopRecording() : startRecording()} style={{marginLeft:8}}>{isRecording ? 'Stop' : 'Record'}</button>
                      <input id="quest-file-input" ref={gameFileRef} type="file" accept="audio/*" style={{display:'none'}} onChange={handleGameFile} />
                    </div>

                    <div style={{marginTop:12}}>
                      <button className={`predict ${gameLoading ? 'working' : ''}`} onClick={handleGamePredict} disabled={gameLoading || !gameFile || !targetEmotion}>{gameLoading ? 'Analyzing…' : 'Analyze My Voice'}</button>
                      <button className="cta" onClick={() => pickNewTarget()} style={{marginLeft:8}}>Next Challenge</button>
                      <button className="reset" onClick={() => { setGameScore(0); setStreak(0); setGameResult(null); }} style={{marginLeft:8}}>Reset Score</button>
                    </div>
                    {isRecording && <div className="recording-indicator" style={{marginTop:8,color:'#ffb27a',fontWeight:800}}>Recording…</div>}
                    {error && <div className="error-msg" style={{marginTop:8,color:'#ff7a7a'}}>{error}</div>}
                  </div>
                </div>

                <div className="quest-right">
                  <div className="score-block">
                    <div className="score-badge">Score: <span className="score-num">{gameScore}</span></div>
                    <div className="streak-badge">Streak: <span className="streak-num">{streak}</span></div>
                  </div>

                  {gameAudioURL && (
                    <div className={`waveform ${gameLoading? 'active':''}`} style={{marginTop:12}}>
                      <span></span><span></span><span></span><span></span><span></span>
                    </div>
                  )}

                  {gameAudioURL && (
                    <div style={{marginTop:12}}>
                      <audio controls src={gameAudioURL} style={{width:'100%'}} />
                      {!isRecording && <div className="muted" style={{marginTop:8}}>Recorded audio ready</div>}
                    </div>
                  )}

                  {gameResult && (
                    <div className="result-card">
                      <div className="result-row"><strong>Target:</strong> <span className="muted">{targetEmotion}</span></div>
                      <div className="result-row"><strong>AI Prediction:</strong> <span className="muted">{gameResult.predicted}</span></div>
                      <div className="result-row"><strong>Confidence:</strong> <span className="muted">{formatConfidence(gameResult.confidence)}%</span></div>
                      <div className="result-row"><strong>Outcome:</strong> <span className={`badge ${gameResult.correct? 'correct':'wrong'}`}>{gameResult.correct ? 'Correct' : 'Missed'}</span></div>

                      <div style={{marginTop:12}} className="muted">Top-3 Predictions</div>
                      <div className="top-bars small">
                        {(gameResult.top_predictions || []).slice(0,3).map((p,i) => (
                          <div className="top-bar" key={i}>
                            <div className="bar-meta"><div className="name">{p.emotion}</div><div className="percent">{formatConfidence(p.confidence)}%</div></div>
                            <div className="bar-track"><div className="bar-fill" style={{width:`${Math.min(100, (p.confidence<=1?p.confidence*100:p.confidence))}%`}} /></div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="about" className="about">
          <div className="container">
            <h2>About the Project</h2>
            <p>This project is an AI-based speech emotion recognition system that analyzes voice recordings and predicts emotional tone from audio patterns.</p>

            <div className="about-grid">
              <div className="about-card card">
                <h3>Voice Processing</h3>
                <p className="muted">Converts speech audio into meaningful voice features.</p>
              </div>
              <div className="about-card card">
                <h3>Emotion Detection</h3>
                <p className="muted">Classifies audio into emotion categories such as happy, sad, angry, calm, and neutral.</p>
              </div>
              <div className="about-card card">
                <h3>Web Experience</h3>
                <p className="muted">Provides an interactive React and Flask-based demo for audio upload and prediction.</p>
              </div>
            </div>
          </div>
        </section>

      </main>

      <footer className="footer">© {new Date().getFullYear()} — Speech Emotion Recognition AI</footer>
    </div>
  )
}
