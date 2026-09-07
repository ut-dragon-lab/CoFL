import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import type { Plugin } from "vite";

/** Include the complete notices for packages whose code reaches the browser. */
export function thirdPartyLicenses(): Plugin {
  return {
    name: "cofl-third-party-licenses",
    generateBundle(_options, bundle) {
      const roots = new Set<string>();
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        for (const [id, info] of Object.entries(output.modules)) {
          if (!info.renderedLength || !id.includes("/node_modules/")) continue;
          let root = dirname(id.replace(/^\0/, "").split("?")[0]);
          while (
            !existsSync(resolve(root, "package.json")) ||
            !JSON.parse(readFileSync(resolve(root, "package.json"), "utf8")).name
          ) {
            const parent = dirname(root);
            if (parent === root) throw new Error(`Package metadata missing: ${id}`);
            root = parent;
          }
          roots.add(root);
        }
      }
      const fallback = JSON.parse(readFileSync(resolve("licenses/sources.json"), "utf8"));
      const sections: string[] = [];
      for (const root of [...roots].sort()) {
        const pkg = JSON.parse(readFileSync(resolve(root, "package.json"), "utf8"));
        const names = readdirSync(root, { withFileTypes: true })
          .filter((entry) => entry.isFile() && /^(licen[sc]e|copying|notice|copyright)(\.|$|-)/i.test(entry.name))
          .map((entry) => entry.name).sort();
        let notices = names.map((name) => `${name}\n${readFileSync(resolve(root, name), "utf8")}`);
        if (!notices.length) {
          const entry = fallback[pkg.name];
          if (!entry || entry.package_version !== pkg.version) {
            throw new Error(`No release license notice for ${pkg.name}@${pkg.version}`);
          }
          notices = [`Source: ${entry.source}\n${readFileSync(resolve("licenses", entry.file), "utf8")}`];
        }
        sections.push(`${pkg.name}@${pkg.version}\nLicense: ${pkg.license ?? "See notice below"}\n\n${notices.join("\n\n")}`);
      }
      this.emitFile({
        type: "asset",
        fileName: "THIRD_PARTY_LICENSES.txt",
        source: "CoFL Studio bundled third-party notices\n\n" + sections.join("\n\n" + "=".repeat(72) + "\n\n"),
      });
    },
  };
}
