/** Bridge Habitat's legacy GOOGLE_texture_basis to Three's bundled upstream
 * Basis Universal WASM codec. GLTFLoader still owns parsing and sampler setup.
 * This adapter contains no texture decoding implementation.
 */
import { DataTexture, FileLoader, Loader } from "three";
import type { GLTFLoader, GLTFLoaderPlugin, GLTFParser } from "three-stdlib";

interface BasisFile {
  getImageWidth(image: number, level: number): number;
  getImageHeight(image: number, level: number): number;
  getImageTranscodedSizeInBytes(
    image: number,
    level: number,
    format: number,
  ): number;
  startTranscoding(): boolean;
  transcodeImage(
    output: Uint8Array,
    image: number,
    level: number,
    format: number,
    unused: number,
    alpha: number,
  ): boolean;
  close(): void;
  delete(): void;
}
interface BasisModule {
  BasisFile: new (bytes: Uint8Array) => BasisFile;
  initializeBasis(): void;
}
let modulePromise: Promise<BasisModule> | undefined;
function codec() {
  modulePromise ??= new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/decoders/basis/basis_transcoder.js";
    script.onload = () => resolve();
    script.onerror = () =>
      reject(new Error("Local Basis decoder is unavailable"));
    document.head.appendChild(script);
  }).then(async () => {
    const factory = (
      globalThis as unknown as {
        BASIS: (options: object) => Promise<BasisModule>;
      }
    ).BASIS;
    const module = await factory({
      locateFile: (name: string) => `/decoders/basis/${name}`,
    });
    module.initializeBasis();
    return module;
  });
  return modulePromise;
}

class HabitatBasisLoader extends Loader<DataTexture> {
  load(
    url: string,
    onLoad: (texture: DataTexture) => void,
    onProgress?: (event: ProgressEvent) => void,
    onError?: (error: unknown) => void,
  ) {
    const texture = new DataTexture();
    new FileLoader(this.manager).setResponseType("arraybuffer").load(
      url,
      async (buffer) => {
        try {
          const module = await codec();
          const file = new module.BasisFile(
            new Uint8Array(buffer as ArrayBuffer),
          );
          try {
            const width = file.getImageWidth(0, 0),
              height = file.getImageHeight(0, 0);
            if (
              !width ||
              !height ||
              width * height > 4096 * 4096 ||
              !file.startTranscoding()
            )
              throw new Error("Invalid or oversized Basis texture");
            const rgba32 = 13; // Binomial's documented cTFRGBA32 transcoder format.
            const bytes = new Uint8Array(
              file.getImageTranscodedSizeInBytes(0, 0, rgba32),
            );
            if (!file.transcodeImage(bytes, 0, 0, rgba32, 0, 0))
              throw new Error("Basis transcoding failed");
            texture.image = { data: bytes, width, height };
            texture.needsUpdate = true;
            onLoad(texture);
          } finally {
            file.close();
            file.delete();
          }
        } catch (error) {
          onError?.(error);
        }
      },
      onProgress,
      onError,
    );
    return texture;
  }
}

export function registerHabitatBasis(loader: GLTFLoader) {
  loader.register(
    (parser: GLTFParser): GLTFLoaderPlugin & { name: string } => ({
      name: "GOOGLE_texture_basis",
      loadTexture(index: number) {
        const extension =
          parser.json.textures[index]?.extensions?.GOOGLE_texture_basis;
        if (!extension) return null;
        return parser
          .loadTextureImage(
            index,
            extension.source,
            new HabitatBasisLoader(parser.options.manager),
          )
          .then((texture) => {
            if (!texture)
              throw new Error("A Habitat Basis texture could not be decoded");
            return texture;
          });
      },
    }),
  );
}
