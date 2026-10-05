import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';

export const fields = ['title', 'question', 'population', 'baseline', 'intervention', 'outcomes', 'assumptions', 'evidence'];
export class Store {
  constructor(root) { this.root = root; this.dir = path.join(root, 'local/workbench/studies'); }
  studyPath(id) {
    if (!/^[a-f0-9-]{36}$/.test(id)) throw new Error('Invalid study ID');
    return path.join(this.dir, id);
  }
  async list() {
    await fs.mkdir(this.dir, { recursive: true });
    const entries = await fs.readdir(this.dir);
    const studies = await Promise.all(entries.map(id => this.read(id).catch(() => null)));
    return studies.filter(Boolean).sort((a,b) => b.updatedAt.localeCompare(a.updatedAt));
  }
  async read(id) { return JSON.parse(await fs.readFile(path.join(this.studyPath(id), 'study.json'), 'utf8')); }
  async save(id, input, revision) {
    const folder = this.studyPath(id);
    let old;
    try { old = await this.read(id); } catch (error) { if (error.code !== 'ENOENT') throw error; }
    if (old && revision !== old.revision) { const error = new Error('Study changed on disk. Reload before saving.'); error.status = 409; throw error; }
    const study = { id, revision: (old?.revision ?? 0) + 1, updatedAt: new Date().toISOString() };
    for (const key of fields) {
      if (typeof input[key] !== 'string' || input[key].length > 100000) throw new Error(`Invalid ${key}`);
      study[key] = input[key];
    }
    await fs.mkdir(folder, { recursive: true });
    const temp = path.join(folder, `study.${randomUUID()}.tmp`);
    await fs.writeFile(temp, JSON.stringify(study, null, 2));
    await fs.rename(temp, path.join(folder, 'study.json'));
    await fs.writeFile(path.join(folder, 'brief.md'), fields.map(key => `## ${key}\n\n${study[key] || '(Unspecified)'}`).join('\n\n') + '\n');
    return study;
  }
  async create(input) { return this.save(randomUUID(), Object.fromEntries(fields.map(k => [k, input[k] ?? ''])), undefined); }
  async safeFile(relative) {
    if (typeof relative !== 'string' || !/^(outputs|local|reports)\//.test(relative)) throw new Error('Artifacts must be under outputs/, local/, or reports/');
    const target = await fs.realpath(path.resolve(this.root, relative));
    const allowed = await Promise.all(['outputs', 'local', 'reports'].map(p => fs.realpath(path.join(this.root,p)).catch(() => path.join(this.root,p))));
    if (!allowed.some(root => target.startsWith(root + path.sep))) throw new Error('Artifact escapes project output folders');
    const stat = await fs.stat(target);
    if (!stat.isFile() || stat.size > 5 * 1024 * 1024) throw new Error('Artifact is not a file or exceeds 5 MB');
    if (!/\.(json|csv|md|txt|yaml|yml|py|html|htm)$/i.test(target)) throw new Error('Unsupported artifact type');
    return target;
  }
  async reports() {
    const reports = []; const warnings = []; let visited = 0;
    const walk = async (folder, depth = 0) => {
      if (depth > 10 || visited++ >= 3000) return;
      let entries;
      try { entries = await fs.readdir(folder, {withFileTypes:true}); } catch { return; }
      for (const entry of entries) {
        const file = path.join(folder, entry.name);
        if (entry.isDirectory()) await walk(file, depth + 1);
        else if (entry.isFile() && /\.html?$/i.test(entry.name)) {
          const relative = path.relative(this.root,file);
          try {
            const safe = await this.safeFile(relative); const stat = await fs.stat(safe);
            reports.push({path:relative, name:entry.name, modified:stat.mtime.toISOString(), size:stat.size});
          } catch (error) { warnings.push(`${relative}: ${error.message}`); }
        }
      }
    };
    for (const folder of ['outputs','local','reports']) await walk(path.join(this.root,folder));
    reports.sort((a,b) => b.modified.localeCompare(a.modified) || a.path.localeCompare(b.path));
    return {reports, warnings, truncated:visited >= 3000};
  }
  async runs() {
    const runs = []; const warnings = []; let visited = 0;
    const walk = async (folder, depth = 0) => {
      if (depth > 10 || visited++ > 3000) return;
      let entries;
      try { entries = await fs.readdir(folder, { withFileTypes: true }); } catch { return; }
      if (entries.some(e => e.name === 'metrics.json' && e.isFile())) {
        const relative = path.relative(this.root,folder);
        const read = async name => {
          try { return JSON.parse(await fs.readFile(await this.safeFile(`${relative}/${name}`), 'utf8')); }
          catch (error) { if (error.code !== 'ENOENT') warnings.push(`${relative}/${name}: ${error.message}`); return null; }
        };
        const [config, metrics, validation] = await Promise.all(['config.json','metrics.json','validation.json'].map(read));
        runs.push({ path: relative, config, metrics, validation,
          artifacts: entries.filter(e => e.isFile() && /\.(json|csv|md|txt|yaml|yml|py|html|htm)$/.test(e.name)).map(e => `${relative}/${e.name}`) });
      }
      for (const entry of entries) if (entry.isDirectory()) await walk(path.join(folder, entry.name), depth + 1);
    };
    await walk(path.join(this.root,'outputs')); await walk(path.join(this.root,'local'));
    return { runs: runs.reverse(), warnings, truncated: visited > 3000 };
  }
}
