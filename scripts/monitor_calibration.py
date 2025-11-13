#!/usr/bin/env python3
"""
Daily calibration monitoring скрипт
Изпълнява се автоматично всеки ден за проверка на калибрацията
"""

import sys
import os
from pathlib import Path

# Добавя root директорията към path
sys.path.insert(0, str(Path(__file__).parent.parent))

from monitoring.adaptive_tuning import AdaptiveTuner
from monitoring.calibration_metrics import CalibrationMonitor
import argparse
import json
from datetime import datetime


def main():
    """
    Главна функция за daily monitoring
    """
    parser = argparse.ArgumentParser(description='Daily Calibration Monitoring')
    parser.add_argument('--days', type=int, default=7, help='Number of days to analyze')
    parser.add_argument('--dry-run', action='store_true', help='Run analysis without applying corrections')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--force', action='store_true', help='Force corrections even if recent ones exist')
    
    args = parser.parse_args()
    
    print(f"🔍 Starting daily calibration monitoring...")
    print(f"   Period: Last {args.days} days")
    print(f"   Dry run: {args.dry_run}")
    print(f"   Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # Инициализира AdaptiveTuner
    tuner = AdaptiveTuner()
    
    if args.force:
        # Временно увеличава лимита за корекции
        tuner.thresholds['max_corrections_per_day'] = 999
    
    # Изпълнява анализа
    try:
        if args.dry_run:
            # Само анализ, без корекции
            analysis = tuner.analyze_calibration_drift(days=args.days)
            
            if 'error' in analysis:
                print(f"❌ Error in analysis: {analysis['error']}")
                return 1
            
            print(f"📊 Analysis Results:")
            print(f"   Matches analyzed: {analysis['n_matches']}")
            print(f"   Issues detected: {len(analysis['issues_detected'])}")
            
            if analysis['issues_detected']:
                print("\n🚨 Issues found:")
                for issue in analysis['issues_detected']:
                    print(f"   - {issue['market']}: {issue['issue']} = {issue['value']:.4f} (threshold: {issue['threshold']:.4f})")
            
            if analysis['recommendations']:
                print(f"\n💡 Recommendations: {len(analysis['recommendations'])}")
                for rec in analysis['recommendations']:
                    print(f"   - {rec['action']}: {rec['description']}")
            
            if not analysis['issues_detected']:
                print("✅ No calibration issues detected!")
        
        else:
            # Пълен мониторинг с корекции
            result = tuner.run_daily_monitoring()
            
            if result['status'] == 'error':
                print(f"❌ Error in monitoring: {result['message']}")
                return 1
            
            analysis = result['analysis']
            corrections_applied = result['corrections_applied']
            
            print(f"📊 Monitoring Results:")
            print(f"   Matches analyzed: {analysis['n_matches']}")
            print(f"   Issues detected: {len(analysis['issues_detected'])}")
            print(f"   Corrections applied: {corrections_applied}")
            
            if analysis['issues_detected']:
                print("\n🚨 Issues found:")
                for issue in analysis['issues_detected']:
                    print(f"   - {issue['market']}: {issue['issue']} = {issue['value']:.4f}")
            
            if corrections_applied:
                print("\n🔧 Automatic corrections applied!")
                print("   Current parameters updated.")
                
                if args.verbose:
                    print("\n📋 Current Parameters:")
                    print(json.dumps(result['current_params'], indent=2))
            
            elif analysis['issues_detected']:
                print("\n⚠️  Issues detected but no corrections applied.")
                print("   (Check recent correction limits or sample size)")
            
            else:
                print("\n✅ No calibration issues detected!")
        
        print("\n" + "=" * 60)
        print("🎉 Daily monitoring completed successfully!")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during monitoring: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def generate_weekly_report():
    """
    Генерира седмичен отчет за калибрацията
    """
    print("📈 Generating weekly calibration report...")
    
    monitor = CalibrationMonitor()
    tuner = AdaptiveTuner()
    
    # Генерира отчет за последната седмица
    report = monitor.generate_calibration_report(days=7)
    
    if 'error' in report:
        print(f"❌ Error generating report: {report['error']}")
        return
    
    # Получава tuning история
    tuning_history = tuner.get_tuning_history(days=7)
    
    # Комбинира в седмичен отчет
    weekly_report = {
        'period': 'Last 7 days',
        'calibration_metrics': report,
        'tuning_history': tuning_history,
        'generated_at': datetime.now().isoformat()
    }
    
    # Запазва седмичния отчет
    weekly_file = f"reports/calibration/weekly_report_{datetime.now().strftime('%Y_W%U')}.json"
    os.makedirs(os.path.dirname(weekly_file), exist_ok=True)
    
    try:
        with open(weekly_file, 'w') as f:
            json.dump(weekly_report, f, indent=2)
        print(f"✅ Weekly report saved: {weekly_file}")
    except Exception as e:
        print(f"❌ Failed to save weekly report: {e}")


if __name__ == "__main__":
    # Проверява дали се изпълнява weekly report
    if len(sys.argv) > 1 and sys.argv[1] == '--weekly-report':
        generate_weekly_report()
    else:
        exit_code = main()
        sys.exit(exit_code)
