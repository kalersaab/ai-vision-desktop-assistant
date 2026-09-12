import type { FormEvent, KeyboardEvent } from 'react'
import type { Message } from './types'

type ConversationPanelProps = {
  messages: Message[]
  draft: string
  busy: boolean
  onDraftChange: (value: string) => void
  onSubmit: () => void
  onReviewProposal: () => void
}

export function ConversationPanel({ messages, draft, busy, onDraftChange, onSubmit, onReviewProposal }: ConversationPanelProps) {
  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      onSubmit()
    }
  }

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    onSubmit()
  }

  return (
    <section className="conversation-panel">
      <div className="panel-heading"><div><span className="section-kicker">CONVERSATION</span><h3>What can I help with?</h3></div><span className="session-id">SESSION / 0042</span></div>
      <div className="timeline" aria-live="polite">
        {messages.map((message) => <article className={`message ${message.role}`} key={message.id}><div className="message-meta"><span>{message.role === 'assistant' ? 'VISION AI' : message.role === 'user' ? 'YOU' : 'SYSTEM'}</span><time>{message.time}</time></div><p>{message.text}</p>{message.proposal && <button className="review-link" onClick={onReviewProposal}>Review proposed actions <span>↗</span></button>}</article>)}
        {busy && <div className="thinking"><span /><span /><span />Analyzing your request</div>}
      </div>
      <form className="composer-wrap" onSubmit={handleSubmit}>
        <div className="composer"><textarea value={draft} onChange={(event) => onDraftChange(event.target.value)} onKeyDown={handleKeyDown} placeholder="Describe an action for your desktop..." rows={1} /><button className="send-button" type="submit" disabled={!draft.trim() || busy} aria-label="Send request">Send <span>↗</span></button></div>
        <span className="composer-hint">Enter to send · Shift + Enter for a new line</span>
      </form>
    </section>
  )
}
