; Inno Setup script: builds PokerSense-Setup.exe from the PyInstaller output.
;
; Expects `dist/PokerSense/` (PyInstaller's COLLECT folder, containing
; PokerSense.exe + dependencies) to already exist -- run the PyInstaller
; build first:
;   pyinstaller packaging/pokersense.spec --distpath dist --workpath build --noconfirm
;   iscc packaging/pokersense.iss
;
; Untested on a real Windows machine (this dev environment is macOS-only) --
; see the "Known-untested" note in pokersense.spec. First real verification
; is CI running on windows-latest, same as everything else Windows-side.

#define MyAppName "PokerSense"
#define MyAppVersion "0.1.11"
#define MyAppExeName "PokerSense.exe"

[Setup]
AppId={{9E9F2C1E-4E1A-4E7B-9C7F-POKERSENSE01}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=PokerSense-Setup
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\PokerSense\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
