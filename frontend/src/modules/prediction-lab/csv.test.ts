import { describe, expect, it } from 'vitest'
import { parseFeatureCsv } from './csv'

describe('prediction feature CSV', () => {
  it('parses used-car space-separated files and keeps empty cells aligned', () => {
    expect(parseFeatureCsv('SaleID model bodyType notRepairedDamage\n150000 1 2 -\n150001  3 0\n')).toEqual([
      { SaleID: 150000, model: 1, bodyType: 2, notRepairedDamage: '-' },
      { SaleID: 150001, model: null, bodyType: 3, notRepairedDamage: '0' },
    ])
  })
  it('preserves numeric-looking mixed categories used during training', () => {
    expect(parseFeatureCsv('id,damage\n1,0.0\n2,1.0\n3,-').map((row) => row.damage)).toEqual(['0.0', '1.0', '-'])
  })
  it('supports BOM, quotes, escaped quotes, embedded newlines and missing numeric values', () => {
    expect(parseFeatureCsv('\uFEFFid,city,age\r\n1,"a,b",\r\n2,"a""b\nc",NaN\r\n')).toEqual([
      { id: 1, city: 'a,b', age: null }, { id: 2, city: 'a"b\nc', age: null },
    ])
  })
  it('rejects misaligned rows and duplicate headers instead of silently losing features', () => {
    expect(() => parseFeatureCsv('id,x\n1,2,3')).toThrow('列数')
    expect(() => parseFeatureCsv('id,id\n1,2')).toThrow('重复')
    expect(() => parseFeatureCsv('id,x\n1,"oops')).toThrow('引号')
  })
})
