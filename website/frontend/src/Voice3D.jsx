import React, { useEffect, useRef } from 'react'

export default function Voice3D({ audioElementId, onLevelChange }) {
  const canvasRef = useRef(null)
  const rafRef = useRef(null)
  const audioCtxRef = useRef(null)
  const analyserRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let width = canvas.width = canvas.clientWidth * devicePixelRatio
    let height = canvas.height = canvas.clientHeight * devicePixelRatio

    function resize() {
      width = canvas.width = canvas.clientWidth * devicePixelRatio
      height = canvas.height = canvas.clientHeight * devicePixelRatio
    }
    window.addEventListener('resize', resize)

    // create or reuse audio context
    const AudioContext = window.AudioContext || window.webkitAudioContext
    if (!AudioContext) {
      ctx.fillStyle = 'rgba(255,255,255,0.03)'
      ctx.fillRect(0, 0, width, height)
      return () => window.removeEventListener('resize', resize)
    }

    const audioEl = document.getElementById(audioElementId)
    let source = null
    if (audioEl) {
      const audioCtx = audioCtxRef.current || new AudioContext()
      audioCtxRef.current = audioCtx
      analyserRef.current = analyserRef.current || audioCtx.createAnalyser()
      analyserRef.current.fftSize = 1024
      try {
        // MediaElementSource can only be created once per element
        source = audioCtx.createMediaElementSource(audioEl)
        source.connect(analyserRef.current)
        analyserRef.current.connect(audioCtx.destination)
      } catch (e) {
        // ignore if already connected
      }
    }

    const bufferLength = analyserRef.current ? analyserRef.current.frequencyBinCount : 512
    const data = new Uint8Array(bufferLength)

    function draw() {
      rafRef.current = requestAnimationFrame(draw)
      if (analyserRef.current) analyserRef.current.getByteFrequencyData(data)

      // compute simple level (0..1)
      if (onLevelChange && analyserRef.current) {
        let sum = 0
        for (let i = 0; i < data.length; i++) sum += data[i]
        const avg = sum / data.length / 255
        try { onLevelChange(avg) } catch(e){}
      }

      ctx.clearRect(0, 0, width, height)

      // background radial vignette
      const g = ctx.createLinearGradient(0, 0, 0, height)
      g.addColorStop(0, 'rgba(5,10,20,0.0)')
      g.addColorStop(1, 'rgba(2,6,12,0.4)')
      ctx.fillStyle = g
      ctx.fillRect(0, 0, width, height)

      // draw radial bars as pseudo-3D voice ring
      const cx = width / 2
      const cy = height / 2
      const radius = Math.min(width, height) * 0.18
      const bars = 64
      for (let i = 0; i < bars; i++) {
        const idx = Math.floor((i / bars) * bufferLength)
        const v = analyserRef.current ? data[idx] / 255 : 0.08
        const angle = (i / bars) * Math.PI * 2
        const x1 = cx + Math.cos(angle) * radius
        const y1 = cy + Math.sin(angle) * radius
        const len = 8 + v * Math.min(width, height) * 0.18
        const x2 = cx + Math.cos(angle) * (radius + len)
        const y2 = cy + Math.sin(angle) * (radius + len)

        // shadow (backface) for 3D feel
        ctx.lineWidth = 6 * devicePixelRatio
        ctx.strokeStyle = `rgba(2,8,20,${0.9 - v * 0.6})`
        ctx.beginPath()
        ctx.moveTo(x1 + 6 * Math.cos(angle + 0.6), y1 + 6 * Math.sin(angle + 0.6))
        ctx.lineTo(x2 + 6 * Math.cos(angle + 0.6), y2 + 6 * Math.sin(angle + 0.6))
        ctx.stroke()

        // main glowing bar
        ctx.lineWidth = 4 * devicePixelRatio
        const grad = ctx.createLinearGradient(x1, y1, x2, y2)
        grad.addColorStop(0, 'rgba(122,252,255,0.02)')
        grad.addColorStop(0.5, 'rgba(155,108,255,0.18)')
        grad.addColorStop(1, 'rgba(122,252,255,0.9)')
        ctx.strokeStyle = grad
        ctx.beginPath()
        ctx.moveTo(x1, y1)
        ctx.lineTo(x2, y2)
        ctx.stroke()
      }

      // center glossy orb
      const orbR = radius * 0.8
      const orbGrad = ctx.createRadialGradient(cx - orbR * 0.2, cy - orbR * 0.3, orbR * 0.08, cx, cy, orbR)
      orbGrad.addColorStop(0, 'rgba(255,255,255,0.9)')
      orbGrad.addColorStop(0.2, 'rgba(122,252,255,0.12)')
      orbGrad.addColorStop(1, 'rgba(6,12,24,0.9)')
      ctx.fillStyle = orbGrad
      ctx.beginPath()
      ctx.arc(cx, cy, orbR, 0, Math.PI * 2)
      ctx.fill()

      // soft rim
      ctx.lineWidth = 2 * devicePixelRatio
      ctx.strokeStyle = 'rgba(255,255,255,0.03)'
      ctx.beginPath(); ctx.arc(cx, cy, orbR + 6 * devicePixelRatio, 0, Math.PI * 2); ctx.stroke()
    }

    draw()

    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', resize)
      try { if (audioCtxRef.current && audioCtxRef.current.state !== 'closed') audioCtxRef.current.close() } catch(e){}
    }
  }, [audioElementId])

  return (
    <div style={{width: '100%', height: '220px'}}>
      <canvas ref={canvasRef} style={{width: '100%', height: '100%', display: 'block', borderRadius: 12}} />
    </div>
  )
}
