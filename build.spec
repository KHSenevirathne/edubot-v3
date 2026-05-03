# PyInstaller spec for EduBot v3
#
# Build a self-contained Windows executable so the application can be
# delivered "without extra installation of libraries", as required by
# the assignment brief.
#
# Run from the project root:
#     pyinstaller build.spec --noconfirm
#
# The result lives at dist/EduBot.exe and bundles:
#   - app.py + the app/ package
#   - templates/ and static/
#   - data/ (intents.json + seeded edubot.db)
#   - models/ (trained chatbot_model.pkl + vectorizer.pkl)

# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
project_root = os.path.abspath(os.getcwd())

a = Analysis(
    ['app.py'],
    pathex=[project_root, os.path.join(project_root, 'app')],
    binaries=[],
    datas=[
        ('templates', 'templates'),
        ('static',    'static'),
        ('data',      'data'),
        ('models',    'models'),
    ],
    hiddenimports=[
        'sklearn.utils._cython_blas',
        'sklearn.neighbors._typedefs',
        'sklearn.tree._utils',
        'sklearn.ensemble._weight_boosting',
        'sklearn.svm._classes',
        'sklearn.naive_bayes',
        'sklearn.feature_extraction.text',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='EduBot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,    # leave True so the seeded model + Flask startup logs are visible
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
