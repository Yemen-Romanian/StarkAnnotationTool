"""Download pretrained STARK-ST weights from the Google Drive folder listed in MODEL_ZOO.md.

The Drive folder is laid out as stark_st2/<variant>/STARKST_ep0050.pth.tar, which maps
onto <save_dir>/checkpoints/train/stark_st2/<variant>/STARKST_ep0050.pth.tar -- exactly
where lib/test/parameter/stark_st.py looks for it.
"""
import argparse
import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

# File ids from the "model" folder linked for STARK-ST50 / STARK-ST101 in MODEL_ZOO.md:
# https://drive.google.com/drive/folders/1fSgll53ZnVKeUn22W37Nijk-b9LGhMdN
VARIANTS = {
    'baseline':                  ('1sV_idlYLyxeCIO2o4AQDvUFO5b-X-E4w', 'STARK-ST50'),
    'baseline_got10k_only':      ('16eK2CxHXYNo3YZ7oEp2FBOojdV1pB0G6', 'STARK-ST50, GOT-10k only'),
    'baseline_R101':             ('1t3oQOF8XyqA3nnMgT4IzzueREEus81uP', 'STARK-ST101'),
    'baseline_R101_got10k_only': ('1IhX1ecuzk8OfAxR74N8BTXG8BE43L3qD', 'STARK-ST101, GOT-10k only'),
}

CKPT_NAME = 'STARKST_ep0050.pth.tar'


def resolve_save_dir():
    """Use the same save_dir the tracker resolves, so the file lands where it is looked for."""
    try:
        from lib.test.evaluation.environment import env_settings
        return env_settings().save_dir
    except Exception as e:
        print('Could not read local.py (%s); falling back to the project root.' % e)
        print('Run setup_env.bat first if the tracker cannot find the checkpoint afterwards.')
        return os.path.realpath(prj_path)


def verify(path):
    """Make sure we got a real checkpoint and not an HTML error page.

    Google Drive answers with HTML (quota exceeded / virus-scan interstitial) instead of
    the file often enough that a silent 3 KB 'checkpoint' is a realistic outcome.
    """
    import torch
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb < 1:
        with open(path, 'rb') as f:
            head = f.read(200)
        raise RuntimeError('Downloaded file is only %.1f KB and starts with %r.\n'
                           'That is almost certainly a Google Drive error page, not a checkpoint.'
                           % (size_mb * 1024, head[:80]))
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if not isinstance(ckpt, dict) or 'net' not in ckpt:
        raise RuntimeError('File loaded but has no "net" entry (keys: %s).'
                           % (list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)))
    n_params = sum(v.numel() for v in ckpt['net'].values() if hasattr(v, 'numel'))
    print('  verified: %.1f MB, %d tensors, %.1fM parameters' % (size_mb, len(ckpt['net']), n_params / 1e6))


def download_variant(variant, save_dir, force=False):
    file_id, desc = VARIANTS[variant]
    out_dir = os.path.join(save_dir, 'checkpoints', 'train', 'stark_st2', variant)
    out_path = os.path.join(out_dir, CKPT_NAME)

    print('\n=== %s (%s) ===' % (variant, desc))
    if os.path.isfile(out_path) and not force:
        print('  already present: %s' % out_path)
        print('  (pass --force to re-download)')
        verify(out_path)
        return out_path

    os.makedirs(out_dir, exist_ok=True)
    try:
        import gdown
    except ImportError:
        raise SystemExit('gdown is not installed. Run:  pip install gdown')

    print('  -> %s' % out_path)
    got = gdown.download(id=file_id, output=out_path, quiet=False)
    if got is None:
        raise RuntimeError(
            'gdown returned no file. Google Drive sometimes rate-limits anonymous downloads.\n'
            'Retry later, or download it by hand from the MODEL_ZOO.md link and save it as:\n  %s'
            % out_path)
    verify(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--variant', default='baseline', choices=sorted(VARIANTS),
                        help='which STARK-ST checkpoint to fetch (default: baseline = STARK-ST50)')
    parser.add_argument('--all', action='store_true', help='fetch every variant')
    parser.add_argument('--force', action='store_true', help='re-download even if the file exists')
    args = parser.parse_args()

    save_dir = resolve_save_dir()
    print('save_dir: %s' % save_dir)

    wanted = sorted(VARIANTS) if args.all else [args.variant]
    done = []
    for variant in wanted:
        done.append(download_variant(variant, save_dir, force=args.force))

    print('\nReady. Annotate a video with:')
    print('  run_annotation.bat <path_to_video>')
    if args.variant != 'baseline' or args.all:
        print('Note: tracker_param must match the variant, e.g.')
        print('  python tracking/annotation_tool.py stark_st %s <video>' % wanted[-1])


if __name__ == '__main__':
    main()
