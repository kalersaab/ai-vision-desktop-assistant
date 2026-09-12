import type { Action } from './types'

type ProposalModalProps = {
  actions: Action[]
  expectedOutcome?: string
  open: boolean
  onResolve: (decision: 'allow' | 'cancel' | 'dry-run') => void
}

export function ProposalModal({ actions, expectedOutcome, open, onResolve }: ProposalModalProps) {
  if (!open) return null

  return (
    <div className="modal-backdrop">
      <section className="proposal-modal" role="dialog" aria-modal="true" aria-labelledby="proposal-title">
        <div className="modal-topline"><span className="risk-label">REVIEW REQUIRED</span><button className="close-button" onClick={() => onResolve('cancel')} aria-label="Close proposal">×</button></div>
        <h3 id="proposal-title">Proposed action sequence</h3>
        <p className="modal-copy">The vision engine identified these steps. Nothing has been executed yet.</p>
        <div className="action-list">{actions.map((action, index) => <div className="action-row" key={`${action.type}-${index}`}><span className="action-index">0{index + 1}</span><strong>{action.type}</strong><span className="action-detail">{action.detail}</span>{action.target && <span className="action-target">{action.target}</span>}</div>)}</div>
        <div className="expected-outcome"><span>EXPECTED OUTCOME</span><strong>{expectedOutcome || 'The requested desktop state is reached.'}</strong></div>
        <div className="modal-actions"><button className="button quiet" onClick={() => onResolve('cancel')}>Cancel</button><button className="button dry" onClick={() => onResolve('dry-run')}>Dry-run</button><button className="button approve" onClick={() => onResolve('allow')}>Allow action <span>↗</span></button></div>
      </section>
    </div>
  )
}
