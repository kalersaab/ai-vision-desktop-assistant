export type Action = { type: string; detail: string; target?: string }

export type Message = {
  id: number
  role: 'system' | 'user' | 'assistant'
  text: string
  time: string
  proposal?: boolean
}
