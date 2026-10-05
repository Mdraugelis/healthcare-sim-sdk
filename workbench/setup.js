// node-pty's macOS prebuild can arrive without executable helper permissions.
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
if (process.platform === 'darwin') {
  const base = path.dirname(fileURLToPath(import.meta.url));
  for (const folder of [`prebuilds/darwin-${process.arch}`, 'build/Release']) {
    try { await fs.chmod(path.join(base,'node_modules/node-pty',folder,'spawn-helper'),0o755); }
    catch (error) { if (error.code !== 'ENOENT') throw error; }
  }
}
