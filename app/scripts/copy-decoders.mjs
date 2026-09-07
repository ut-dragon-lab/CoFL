import { cp, mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const three = resolve(
  dirname(fileURLToPath(import.meta.resolve("three"))),
  "..",
);
const output = resolve("public/decoders");
await mkdir(output, { recursive: true });
for (const name of ["basis", "draco"]) {
  await cp(resolve(three, "examples/jsm/libs", name), resolve(output, name), {
    recursive: true,
  });
}
await cp(resolve(three, "LICENSE"), resolve(output, "THREE-LICENSE.txt"));
await cp(resolve("licenses/basis_universal-LICENSE.txt"), resolve(output, "basis/LICENSE.txt"));
await cp(resolve("licenses/draco3d-LICENSE.txt"), resolve(output, "draco/LICENSE.txt"));
