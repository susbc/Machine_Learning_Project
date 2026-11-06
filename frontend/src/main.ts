import { createApp } from 'vue'

import App from './App.vue'
import { elementPlusComponents } from './app/element-plus'
import { createAppPinia } from './app/pinia'
import { createAppRouter } from './app/router'
import './styles/main.css'
import './styles/workspace-theme.css'

const app = createApp(App)

app.use(createAppPinia())
app.use(createAppRouter())
elementPlusComponents.forEach((component) => app.use(component))

app.mount('#app')
