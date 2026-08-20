import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import { LevelProvider } from './components/LevelContext'
import './styles.css'
import './styles-learn.css'
import './styles-player.css'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LevelProvider>
      <App />
    </LevelProvider>
  </StrictMode>,
)
