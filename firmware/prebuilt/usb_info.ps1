# In thong tin may + USB de so sanh 2 laptop (may flash duoc vs may loi). Chi DOC, khong sua gi.
#   powershell -ExecutionPolicy Bypass -File usb_info.ps1
$cs = Get-CimInstance Win32_ComputerSystem
$bios = Get-CimInstance Win32_BIOS
$os = Get-CimInstance Win32_OperatingSystem
"Model        : $($cs.Manufacturer) $($cs.Model) ($((Get-CimInstance Win32_ComputerSystemProduct).Version))"
"BIOS         : $($bios.SMBIOSBIOSVersion)  ($($bios.ReleaseDate.ToString('yyyy-MM-dd')))"
"Windows      : $($os.Caption) build $($os.BuildNumber).$((Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').UBR)"
"Power plan   : $((powercfg /getactivescheme) -replace '.*\((.*)\).*','$1')"
foreach ($d in Get-PnpDevice -PresentOnly | Where-Object { $_.FriendlyName -match 'Host Controller|Root Hub|CH343|CH34|USB Serial Device|USB4|Thunderbolt' }) {
    $v = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_Device_DriverVersion -ErrorAction SilentlyContinue).Data
    $p = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_Device_DriverProvider -ErrorAction SilentlyContinue).Data
    $l = (Get-PnpDeviceProperty -InstanceId $d.InstanceId -KeyName DEVPKEY_Device_LocationInfo -ErrorAction SilentlyContinue).Data
    "USB          : $($d.FriendlyName) | driver $v ($p) $l"
}
$vantage = Get-AppxPackage -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'LenovoCompanion|LenovoVantage|LegionSpace|LegionZone' } | Select-Object -First 1
if ($vantage) { "Lenovo app   : $($vantage.Name) $($vantage.Version)" }
"Antivirus    : " + ((Get-CimInstance -Namespace root/SecurityCenter2 -ClassName AntiVirusProduct -ErrorAction SilentlyContinue).displayName -join ', ')
