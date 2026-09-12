export function ControlPlane() {
  return (
    <aside className="inspector-panel">
      <div className="inspector-header"><span className="section-kicker">CONTROL PLANE</span><span className="live-tag">● LIVE</span></div>
      <div className="screen-preview"><div className="preview-grid" /><div className="preview-label">SCREEN OBSERVATION</div><div className="preview-crosshair">+</div><div className="preview-window"><div /><div /><div /></div></div>
      <div className="inspector-row"><span>Vision state</span><strong><i className="green-dot" />Watching</strong></div>
      <div className="inspector-row"><span>Resolution</span><strong>1280 × 720</strong></div>
      <div className="inspector-row"><span>Safety policy</span><strong>Strict</strong></div>
      <div className="inspector-divider" />
      <div className="next-step"><span className="section-kicker">NEXT STEP</span><p>Every action pauses here for your approval.</p></div>
    </aside>
  )
}
