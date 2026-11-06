/** Parse CSV/TSV/space-separated feature files, including quoted newlines. */
function readCells(text: string, delimiter: string): string[][] {
  const records: string[][] = []
  let row: string[] = []
  let cell = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const char = text[i]
    if (char === '"') {
      if (quoted && text[i + 1] === '"') { cell += '"'; i++ }
      else quoted = !quoted
    } else if (!quoted && char === delimiter) {
      row.push(cell.trim()); cell = ''
    } else if (!quoted && (char === '\n' || char === '\r')) {
      row.push(cell.trim()); records.push(row); row = []; cell = ''
      if (char === '\r' && text[i + 1] === '\n') i++
    } else cell += char
  }
  if (quoted) throw new Error('CSV 引号未闭合，请检查文件格式')
  if (cell || row.length) { row.push(cell.trim()); records.push(row) }
  return records.filter((record) => record.some((value) => value !== ''))
}

const isMissing = (value: string) => value === '' || /^(nan|null|na)$/i.test(value)

export function parseFeatureCsv(raw: string): Record<string, unknown>[] {
  const text = raw.replace(/^\uFEFF/, '')
  const headerLine = text.split(/\r?\n/, 1)[0] ?? ''
  const delimiter = [',', '\t', ';', '|', ' '].reduce((best, candidate) =>
    (readCells(headerLine, candidate)[0]?.length ?? 0) > (readCells(headerLine, best)[0]?.length ?? 0) ? candidate : best, ',')
  const [headers, ...records] = readCells(text, delimiter)
  if (!headers || !records.length) return []
  if (headers.some((name) => !name) || new Set(headers).size !== headers.length) {
    throw new Error('CSV 表头存在空字段或重复字段')
  }
  records.forEach((record, index) => {
    if (record.length !== headers.length) throw new Error(`CSV 第 ${index + 2} 条记录的列数与表头不一致`)
  })
  // Infer types per column, as the training CSV reader does. Mixed categorical
  // values such as "0.0", "1.0", "-" must retain their exact category labels.
  const types = headers.map((_, column) => {
    const values = records.map((record) => record[column]!).filter((value) => !isMissing(value))
    if (values.every((value) => Number.isFinite(Number(value)))) return 'number'
    if (values.every((value) => /^(true|false)$/i.test(value))) return 'boolean'
    return 'string'
  })
  return records.map((record) => Object.fromEntries(headers.map((name, column) => {
    const value = record[column]!
    return [name, isMissing(value) ? null : types[column] === 'number' ? Number(value)
      : types[column] === 'boolean' ? value.toLowerCase() === 'true' : value]
  })))
}
