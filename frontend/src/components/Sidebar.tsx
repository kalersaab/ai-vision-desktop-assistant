export function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="brand-mark">AV</div>
      <div className="brand-copy"><span className="eyebrow">PRIVATE DESKTOP AGENT</span><h1>Vision Desk</h1></div>
      <div className="sidebar-divider" />
      <nav className="nav-list" aria-label="Workspace navigation">
        <button className="nav-item active"><span className="nav-dot" />Live session</button>
        <button className="nav-item"><span className="nav-icon">◌</span>Activity log</button>
        <button className="nav-item"><span className="nav-icon">⌘</span>Settings</button>
      </nav>
      <div className="sidebar-footer"><div className="status-line"><span className="status-pulse" />Local engine online</div><div className="engine-version">Qwen vision · C++ acceleration</div></div>
    </aside>
  )
}
