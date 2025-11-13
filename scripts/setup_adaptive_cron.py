#!/usr/bin/env python3
"""
Setup script за автоматично стартиране на adaptive learning
"""

import sys
import os
from pathlib import Path
import subprocess
import platform

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.utils import setup_logging


def get_project_root():
    """Получава root директорията на проекта"""
    return str(Path(__file__).parent.parent.absolute())


def create_cron_job():
    """Създава cron job за Linux/macOS"""
    try:
        project_root = get_project_root()
        python_path = sys.executable
        script_path = os.path.join(project_root, "scripts/performance_monitor.py")
        
        # Cron entry - всяка неделя в 3:00 AM
        cron_entry = f"0 3 * * 0 cd {project_root} && {python_path} {script_path} >> logs/adaptive_cron.log 2>&1"
        
        print("🕐 Създаване на cron job...")
        print(f"Entry: {cron_entry}")
        
        # Получава текущия crontab
        try:
            current_crontab = subprocess.check_output(['crontab', '-l'], stderr=subprocess.DEVNULL).decode()
        except subprocess.CalledProcessError:
            current_crontab = ""
        
        # Проверява дали entry-то вече съществува
        if "performance_monitor.py" in current_crontab:
            print("⚠️ Cron job вече съществува")
            return True
        
        # Добавя новия entry
        new_crontab = current_crontab + cron_entry + "\n"
        
        # Записва новия crontab
        process = subprocess.Popen(['crontab', '-'], stdin=subprocess.PIPE)
        process.communicate(input=new_crontab.encode())
        
        if process.returncode == 0:
            print("✅ Cron job създаден успешно")
            return True
        else:
            print("❌ Грешка при създаване на cron job")
            return False
            
    except Exception as e:
        print(f"❌ Грешка при създаване на cron job: {e}")
        return False


def create_launchd_job():
    """Създава LaunchAgent за macOS"""
    try:
        project_root = get_project_root()
        python_path = sys.executable
        script_path = os.path.join(project_root, "scripts/performance_monitor.py")
        
        # LaunchAgent plist съдържание
        plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.footballai.adaptive.learning</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{project_root}</string>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Weekday</key>
        <integer>0</integer>
        <key>Hour</key>
        <integer>3</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>{project_root}/logs/adaptive_launchd.log</string>
    <key>StandardErrorPath</key>
    <string>{project_root}/logs/adaptive_launchd_error.log</string>
</dict>
</plist>"""
        
        # Пътя към LaunchAgents
        home_dir = os.path.expanduser("~")
        launchagents_dir = os.path.join(home_dir, "Library/LaunchAgents")
        plist_path = os.path.join(launchagents_dir, "com.footballai.adaptive.learning.plist")
        
        # Създава директорията ако не съществува
        os.makedirs(launchagents_dir, exist_ok=True)
        
        # Записва plist файла
        with open(plist_path, 'w') as f:
            f.write(plist_content)
        
        print(f"📄 LaunchAgent plist създаден: {plist_path}")
        
        # Зарежда LaunchAgent
        result = subprocess.run(['launchctl', 'load', plist_path], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ LaunchAgent зареден успешно")
            return True
        else:
            print(f"❌ Грешка при зареждане на LaunchAgent: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Грешка при създаване на LaunchAgent: {e}")
        return False


def create_windows_task():
    """Създава Windows Task Scheduler задача"""
    try:
        project_root = get_project_root()
        python_path = sys.executable
        script_path = os.path.join(project_root, "scripts\\performance_monitor.py")
        
        # Windows Task Scheduler команда
        task_name = "FootballAI_Adaptive_Learning"
        
        # XML за задачата
        task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2023-01-01T03:00:00</StartBoundary>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek>
          <Sunday />
        </DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions>
    <Exec>
      <Command>{python_path}</Command>
      <Arguments>{script_path}</Arguments>
      <WorkingDirectory>{project_root}</WorkingDirectory>
    </Exec>
  </Actions>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>false</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <IdleSettings>
      <StopOnIdleEnd>true</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT72H</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
</Task>"""
        
        # Записва XML файла
        xml_path = os.path.join(project_root, "adaptive_task.xml")
        with open(xml_path, 'w', encoding='utf-16') as f:
            f.write(task_xml)
        
        # Създава задачата
        result = subprocess.run([
            'schtasks', '/create', '/tn', task_name, 
            '/xml', xml_path, '/f'
        ], capture_output=True, text=True)
        
        # Изтрива временния XML файл
        os.remove(xml_path)
        
        if result.returncode == 0:
            print("✅ Windows Task създадена успешно")
            return True
        else:
            print(f"❌ Грешка при създаване на Windows Task: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Грешка при създаване на Windows Task: {e}")
        return False


def setup_logging_directory():
    """Създава директория за логове"""
    project_root = get_project_root()
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    print(f"📁 Logs директория: {logs_dir}")


def main():
    """Главна функция за setup на автоматично стартиране"""
    print("🤖 SETUP НА ADAPTIVE LEARNING AUTOMATION")
    print("=" * 60)
    
    # Създава logs директория
    setup_logging_directory()
    
    # Определя операционната система
    system = platform.system().lower()
    
    print(f"🖥️ Операционна система: {system}")
    print(f"📂 Проект директория: {get_project_root()}")
    print(f"🐍 Python path: {sys.executable}")
    
    success = False
    
    if system == "linux":
        print("\n🐧 Настройка за Linux с cron...")
        success = create_cron_job()
        
    elif system == "darwin":  # macOS
        print("\n🍎 Настройка за macOS...")
        print("Избери метод:")
        print("1. Cron job (препоръчително)")
        print("2. LaunchAgent")
        
        choice = input("Избор (1/2): ").strip()
        
        if choice == "2":
            success = create_launchd_job()
        else:
            success = create_cron_job()
            
    elif system == "windows":
        print("\n🪟 Настройка за Windows с Task Scheduler...")
        success = create_windows_task()
        
    else:
        print(f"❌ Неподдържана операционна система: {system}")
        return False
    
    if success:
        print("\n✅ Автоматизацията е настроена успешно!")
        print("\n📋 Следващи стъпки:")
        print("1. Провери че adaptive learning е enabled в config/adaptive_config.yaml")
        print("2. Тествай ръчно: python3 scripts/performance_monitor.py")
        print("3. Провери логовете в logs/ директорията")
        
        if system in ["linux", "darwin"]:
            print("4. Провери cron jobs: crontab -l")
        elif system == "windows":
            print("4. Провери Task Scheduler за 'FootballAI_Adaptive_Learning'")
        
        print("\n🕐 Автоматичното стартиране ще се случи всяка неделя в 3:00 AM")
        
    else:
        print("\n❌ Грешка при настройка на автоматизацията")
        print("\n🔧 Ръчна настройка:")
        print("За Linux/macOS добави в crontab:")
        print(f"0 3 * * 0 cd {get_project_root()} && {sys.executable} scripts/performance_monitor.py")
        
    return success


if __name__ == "__main__":
    main()
