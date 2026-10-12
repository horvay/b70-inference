"""Pull an image by digest from a registry and unpack it into a root filesystem, without Docker.

    python3 oci_pull.py <registry> <repository> <digest> <rootfs dir> [--plan]

Layers are applied in order, honouring OCI whiteouts (.wh.<name>, .wh..wh..opq). Blobs are cached in
<rootfs>/../blobs and verified against their sha256, so an interrupted pull resumes.
"""
import gzip, hashlib, io, json, os, shutil, sys, tarfile, urllib.error, urllib.request

REG, REPO, DIGEST, ROOT = sys.argv[1:5]
ROOT = os.path.abspath(ROOT)
PLAN = "--plan" in sys.argv
BLOBS = os.path.join(os.path.dirname(os.path.abspath(ROOT)), "blobs")

def token():
    if REG == "registry-1.docker.io":
        url = f"https://auth.docker.io/token?service=registry.docker.io&scope=repository:{REPO}:pull"
    else:
        url = f"https://{REG}/token?scope=repository:{REPO}:pull"
    return json.load(urllib.request.urlopen(url))["token"]

TOK = token()
ACCEPT = ", ".join(["application/vnd.oci.image.index.v1+json", "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.list.v2+json",
                    "application/vnd.docker.distribution.manifest.v2+json"])

def get(path, accept=ACCEPT):
    req = urllib.request.Request(f"https://{REG}/v2/{REPO}/{path}",
                                 headers={"Authorization": f"Bearer {TOK}", "Accept": accept})
    return urllib.request.urlopen(req)

def manifest(ref):
    return json.load(get(f"manifests/{ref}"))

m = manifest(DIGEST)
if "manifests" in m:   # an index: the linux/amd64 image
    m = manifest(next(x["digest"] for x in m["manifests"]
                      if x.get("platform", {}).get("os") == "linux" and x["platform"].get("architecture") == "amd64"))
layers = m["layers"]
total = sum(l["size"] for l in layers)
print(f"{len(layers)} layers, {total / 1e9:.2f} GB compressed")
if PLAN:
    sys.exit(0)

os.makedirs(BLOBS, exist_ok=True)
os.makedirs(ROOT, exist_ok=True)

def fetch(layer):
    """Download a blob into the cache, resuming a partial file with a Range request; a stalled connection (no
    bytes for 60 s) is dropped and retried from where it stopped."""
    import socket, time
    d = layer["digest"]; path = os.path.join(BLOBS, d.replace(":", "_"))
    if os.path.exists(path) and os.path.getsize(path) == layer["size"]:
        return path
    tmp = path + ".part"
    for attempt in range(50):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if have >= layer["size"]:
            break
        global TOK
        try:
            req = urllib.request.Request(f"https://{REG}/v2/{REPO}/blobs/{d}",
                                         headers={"Authorization": f"Bearer {TOK}", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "ab" if have else "wb") as f:
                if have and r.status != 206:      # the server ignored the range: start over
                    f.truncate(0)
                while chunk := r.read(1 << 22):
                    f.write(chunk)
        except urllib.error.HTTPError as e:
            if e.code == 401:                      # registry tokens expire after a few minutes: get a new one
                TOK = token()
                continue
            print(f"  {d[:19]}: HTTP {e.code}; retrying", flush=True)
            time.sleep(min(30, 2 + attempt))
        except (OSError, socket.timeout) as e:
            print(f"  {d[:19]}: {e}; retrying from {os.path.getsize(tmp) if os.path.exists(tmp) else 0} bytes",
                  flush=True)
            time.sleep(min(30, 2 + attempt))
    h = hashlib.sha256()
    with open(tmp, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    if "sha256:" + h.hexdigest() != d:
        os.unlink(tmp)
        raise SystemExit(f"digest mismatch for {d}")
    os.rename(tmp, path)
    return path

def open_layer(path):
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"\x1f\x8b":
        return tarfile.open(path, "r:gz")
    if magic == b"\x28\xb5\x2f\xfd":
        from compression import zstd
        return tarfile.open(fileobj=zstd.open(path, "rb"), mode="r|")
    return tarfile.open(path, "r:")

def safe(name):
    p = os.path.normpath(os.path.join(ROOT, name.lstrip("/")))
    if not (p == os.path.abspath(ROOT) or p.startswith(os.path.abspath(ROOT) + os.sep)):
        raise SystemExit(f"path escapes rootfs: {name}")
    return p

def remove(p):
    if os.path.islink(p) or os.path.isfile(p):
        os.unlink(p)
    elif os.path.isdir(p):
        shutil.rmtree(p)

for i, layer in enumerate(layers):
    path = fetch(layer)
    print(f"layer {i + 1}/{len(layers)} {layer['size'] / 1e6:.0f} MB", flush=True)
    with open_layer(path) as tf:
        for member in tf:
            name = member.name
            base = os.path.basename(name); parent = os.path.dirname(name)
            if base == ".wh..wh..opq":           # opaque dir: clear what lower layers put there
                d = safe(parent)
                if os.path.isdir(d):
                    for e in os.listdir(d):
                        remove(os.path.join(d, e))
                continue
            if base.startswith(".wh."):
                remove(safe(os.path.join(parent, base[4:])))
                continue
            dest = safe(name)
            if os.path.lexists(dest) and not (member.isdir() and os.path.isdir(dest) and not os.path.islink(dest)):
                remove(dest)
            if member.ischr() or member.isblk() or member.isfifo():
                continue                         # device nodes: bwrap provides /dev
            if member.islnk():
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                os.link(safe(member.linkname), dest)
                continue
            member.mode |= 0o200                 # user-writable, so later layers can replace it
            tf.extract(member, ROOT, set_attrs=False, filter="fully_trusted") if not member.issym() else (
                os.makedirs(os.path.dirname(dest), exist_ok=True), os.symlink(member.linkname, dest))
            if member.isfile() or member.isdir():
                os.chmod(dest, (member.mode & 0o7777) | 0o200)
print("rootfs ready:", ROOT)
