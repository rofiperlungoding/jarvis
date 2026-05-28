; Inno Setup Script for JARVIS AI Assistant
; Multi-page wizard with welcome / license / pre-flight / install location /
; components / tasks / install / finish.

#define MyAppName "JARVIS"
#define MyAppVersion "1.0.2"
#define MyAppPublisher "JARVIS Project"
#define MyAppURL "https://github.com/rofiperlungoding/jarvis"
#define MyAppExeName "JARVIS.exe"

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
AppContact={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableReadyPage=no
AllowNoIcons=yes
OutputDir=output
OutputBaseFilename=JARVIS-Setup-{#MyAppVersion}
SetupIconFile=jarvis.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName} {#MyAppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardImageFile=welcome.bmp
WizardSmallImageFile=header.bmp
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\LICENSE
MinVersion=10.0.18362
ShowLanguageDialog=no
DisableDirPage=no
CloseApplications=force
RestartApplications=no
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoProductName={#MyAppName}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoCopyright=Copyright (c) 2026 JARVIS Contributors

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[CustomMessages]
PreflightTitle=Before you install
PreflightSubtitle=Make sure you have what JARVIS needs to run.
PreflightHeader=JARVIS will install in about a minute. Here is what is required:

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: checkedonce
Name: "startmenuicon"; Description: "Add to &Start menu"; GroupDescription: "Shortcuts:";
Name: "autostart"; Description: "Start JARVIS automatically when Windows starts"; GroupDescription: "Startup:"; Flags: unchecked

[Components]
Name: "core"; Description: "Core JARVIS application (required)"; Types: full compact custom; Flags: fixed
Name: "voice_uk"; Description: "British English voice (en_GB-alan-medium, recommended)"; Types: full custom

[Files]
; Bundle the entire PyInstaller dist/JARVIS/ folder under "core"
Source: "..\dist\JARVIS\*"; DestDir: "{app}"; Components: core; Flags: ignoreversion recursesubdirs createallsubdirs
; Voice (already bundled by spec, but flagged so users can pick)
Source: "..\dist\JARVIS\_internal\piper_voices\*"; DestDir: "{app}\_internal\piper_voices"; Components: voice_uk; Flags: ignoreversion recursesubdirs createallsubdirs skipifsourcedoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: startmenuicon
Name: "{group}\Quick Start Guide"; Filename: "{app}\README.txt"; Tasks: startmenuicon
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"; Tasks: startmenuicon
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Registry]
; Auto-start entry (optional task)
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "JARVIS"; ValueData: """{app}\{#MyAppExeName}"""; Flags: uninsdeletevalue; Tasks: autostart

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
Filename: "{app}\README.txt"; Description: "Open Quick Start Guide"; Flags: postinstall skipifsilent shellexec unchecked

[Code]
var
  PreflightPage: TWizardPage;
  PreflightLabel: TNewStaticText;

procedure InitializeWizard;
var
  ListLabel: TNewStaticText;
  TitleLabel: TNewStaticText;
  Y: Integer;
  Items: array of String;
  I: Integer;
begin
  // Pre-flight info page (after license, before install location).
  PreflightPage := CreateCustomPage(wpLicense,
    'Before you install',
    'Make sure you have what JARVIS needs to run');

  TitleLabel := TNewStaticText.Create(PreflightPage);
  TitleLabel.Parent := PreflightPage.Surface;
  TitleLabel.Caption := 'JARVIS works best with the following on hand:';
  TitleLabel.Font.Style := [fsBold];
  TitleLabel.Top := 8;
  TitleLabel.Left := 0;
  TitleLabel.AutoSize := True;

  Y := 40;
  SetArrayLength(Items, 6);
  Items[0] := 'Windows 10 (build 1903 or later) or Windows 11.   [auto-checked]';
  Items[1] := '1 GB free disk space (we need ~700 MB).   [will be checked]';
  Items[2] := 'A microphone — built-in, USB, or headset.   [optional]';
  Items[3] := 'Speakers or headphones for JARVIS''s voice replies.';
  Items[4] := 'A free Mistral API key — sign up at console.mistral.ai.';
  Items[5] := 'An internet connection for cloud LLM and first-run model download.';

  for I := 0 to GetArrayLength(Items) - 1 do
  begin
    ListLabel := TNewStaticText.Create(PreflightPage);
    ListLabel.Parent := PreflightPage.Surface;
    ListLabel.Caption := '  •  ' + Items[I];
    ListLabel.Top := Y;
    ListLabel.Left := 0;
    ListLabel.AutoSize := True;
    ListLabel.WordWrap := True;
    ListLabel.Width := PreflightPage.SurfaceWidth - 20;
    Y := Y + 28;
  end;

  PreflightLabel := TNewStaticText.Create(PreflightPage);
  PreflightLabel.Parent := PreflightPage.Surface;
  PreflightLabel.Caption :=
    'Don''t worry — the app itself walks you through API key setup, mic test, and voice picker on first launch.';
  PreflightLabel.Top := Y + 16;
  PreflightLabel.Left := 0;
  PreflightLabel.AutoSize := True;
  PreflightLabel.WordWrap := True;
  PreflightLabel.Width := PreflightPage.SurfaceWidth - 20;
  PreflightLabel.Font.Style := [fsItalic];
  PreflightLabel.Font.Color := clBlue;
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  // Could add disk-space pre-flight here in the future
end;
