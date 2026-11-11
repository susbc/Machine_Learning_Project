import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { defineComponent, h, ref, resolveComponent } from 'vue'

import { elementPlusComponents } from './element-plus'

const SRC_DIR = join(process.cwd(), 'src')

function vueFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((entry) => {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) return vueFiles(path)
    return entry.endsWith('.vue') ? [path] : []
  })
}

function toComponentName(tag: string) {
  return `El${tag.split('-').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join('')}`
}

/** 模拟 app.use：收集 install 真正注册的组件名（含 withInstall 附带注册的子组件）。 */
function registeredNames(): Set<string> {
  const names = new Set<string>()
  const recorder = {
    component: (name: string) => {
      names.add(name)
    },
  }
  for (const plugin of elementPlusComponents) {
    const install = (plugin as { install?: (app: typeof recorder) => void }).install
    install?.(recorder)
  }
  return names
}

describe('element plus on-demand registry', () => {
  it('registers every el- component used by templates', () => {
    const used = new Set<string>()
    for (const file of vueFiles(SRC_DIR)) {
      const source = readFileSync(file, 'utf-8')
      for (const match of source.matchAll(/<el-([a-z][a-z0-9]*(?:-[a-z0-9]+)*)/g)) {
        used.add(toComponentName(match[1]!))
      }
    }
    const registered = registeredNames()

    expect([...used].sort()).toEqual([...used].filter((name) => registered.has(name)).sort())
    expect(used.size).toBeGreaterThan(0)
    // withNoopInstall 陷阱防护：radio 家族必须由 ElRadio 的 install 附带注册
    expect([...registered]).toEqual(expect.arrayContaining(['ElRadio', 'ElRadioGroup', 'ElRadioButton']))
  })

  it('renders radio buttons with the on-demand registry', () => {
    const Demo = defineComponent({
      setup: () => {
        const value = ref('file')
        return () =>
          h(
            resolveComponent('el-radio-group'),
            { modelValue: value.value, 'onUpdate:modelValue': (next: string) => { value.value = next } },
            {
              default: () => [
                h(resolveComponent('el-radio-button'), { value: 'file' }, { default: () => '服务器文件' }),
                h(resolveComponent('el-radio-button'), { value: 'database' }, { default: () => '数据库' }),
              ],
            },
          )
      },
    })
    const wrapper = mount(Demo, { global: { plugins: elementPlusComponents } })

    expect(wrapper.findAll('.el-radio-button')).toHaveLength(2)
    expect(wrapper.text()).toContain('服务器文件')
    expect(wrapper.text()).toContain('数据库')
    wrapper.unmount()
  })

  it('imports scrollbar styles required by select and table scroll containers', () => {
    const source = readFileSync(join(SRC_DIR, 'app', 'element-plus.ts'), 'utf-8')

    expect(source).toContain("import 'element-plus/theme-chalk/el-scrollbar.css'")
  })
})
