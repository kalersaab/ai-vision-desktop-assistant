import { useEffect, useRef, useState } from 'react'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import './App.css'
import { ConversationPanel, ControlPlane, ProposalModal, Sidebar, Topbar } from './components'
import type { Action, Message } from './components'

type AgentEvent = {
  id?: string
  event: string
  message?: string
  plan?: {
    actions?: Array<{ type: string; target?: string; x?: number; y?: number; text?: string; key?: string; duration_ms?: number }>
    expected_outcome?: string
  }
}

type BackendAction = {
  type: string
  target?: string
  x?: number
  y?: number
  text?: string
  key?: string
  duration_ms?: number
}

const initialMessages: Message[] = [
  { id: 1, role: 'system', text: 'Vision engine ready. Your desktop is being observed locally.', time: '09:41' },
  { id: 2, role: 'assistant', text: 'Tell me what you want to do. I will show the complete action plan before anything touches your computer.', time: '09:41' },
]

function formatAction(action: BackendAction): Action {
  let detail = ''
  if (action.type === 'type' && action.text) detail = `"${action.text}"`
  else if (action.type === 'press' && action.key) detail = action.key
  else if (action.type === 'wait' && action.duration_ms !== undefined) detail = `${action.duration_ms}ms`
  else if (action.x !== undefined && action.y !== undefined) detail = `(${action.x}, ${action.y})`
  return { type: action.type.toUpperCase(), detail, target: action.target }
}

function App() {
  const [messages, setMessages] = useState(initialMessages)
  const [draft, setDraft] = useState('')
  const [proposalOpen, setProposalOpen] = useState(false)
  const [proposalActions, setProposalActions] = useState<Action[]>([])
  const [expectedOutcome, setExpectedOutcome] = useState('')
  const [busy, setBusy] = useState(false)
  const [bridgeError, setBridgeError] = useState('')
  const requestId = useRef('')
  const nextMessageId = useRef(initialMessages.length + 1)

  const createMessageId = () => {
    const id = nextMessageId.current
    nextMessageId.current += 1
    return id
  }

  useEffect(() => {
    let unlisten: UnlistenFn | undefined
    const setup = async () => {
      try {
        unlisten = await listen<AgentEvent>('agent-event', ({ payload }) => {
          if (payload.id && payload.id !== requestId.current) return
          if (payload.event === 'started') setBusy(true)
          if (payload.event === 'status') setMessages((current) => [...current, { id: createMessageId(), role: 'system', text: payload.message || 'Working...', time: 'now' }])
          if (payload.event === 'proposal' && payload.plan) {
            setProposalActions((payload.plan.actions || []).map(formatAction))
            setExpectedOutcome(payload.plan.expected_outcome || '')
            setProposalOpen(true)
          }
          if (['success', 'failed', 'blocked', 'cancelled', 'error'].includes(payload.event)) {
            setMessages((current) => [...current, { id: createMessageId(), role: payload.event === 'success' ? 'assistant' : 'system', text: payload.message || payload.event, time: 'now' }])
            setBusy(false)
            setProposalOpen(false)
          }
          if (payload.event === 'bridge-stopped') {
            setBusy(false)
            setBridgeError('Python backend stopped. Restart the Tauri app to reconnect.')
          }
          if (payload.event === 'result') setBusy(false)
        })
        await invoke('start_agent')
      } catch (error) {
        setBridgeError(String(error))
      }
    }
    void setup()
    return () => { unlisten?.(); void invoke('stop_agent').catch(() => undefined) }
  }, [])

  const submitRequest = async () => {
    const request = draft.trim()
    if (!request || busy) return
    const id = `request-${createMessageId()}`
    requestId.current = id
    setDraft('')
    setBusy(true)
    setMessages((current) => [...current, { id: createMessageId(), role: 'user', text: request, time: 'now' }])
    try {
      await invoke('send_agent_message', { message: { id, type: 'request', request } })
    } catch (error) {
      setBridgeError(String(error))
      setBusy(false)
    }
  }

  const resolveProposal = async (decision: 'allow' | 'cancel' | 'dry-run') => {
    setProposalOpen(false)
    try {
      await invoke('send_agent_message', { message: { id: requestId.current, type: 'confirm', decision: decision === 'allow' ? 'y' : decision === 'dry-run' ? 'dry-run' : 'n' } })
    } catch (error) {
      setBridgeError(String(error))
      setBusy(false)
    }
  }

  return (
    <main className="app-shell">
      <Sidebar />
      <section className="workspace">
        <Topbar />
        {bridgeError && <div className="bridge-error" role="status">{bridgeError}</div>}
        <div className="session-grid">
          <ConversationPanel messages={messages} draft={draft} busy={busy} onDraftChange={setDraft} onSubmit={() => void submitRequest()} onReviewProposal={() => setProposalOpen(true)} />
          <ControlPlane />
        </div>
      </section>
      <ProposalModal actions={proposalActions} expectedOutcome={expectedOutcome} open={proposalOpen} onResolve={(decision) => void resolveProposal(decision)} />
    </main>
  )
}

export default App
