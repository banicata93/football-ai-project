"""
Тестове за новата dependency injection архитектура
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient

from api.main import app
from core.service_manager import get_service_manager
from api.prediction_service import PredictionService
from api.improved_prediction_service import ImprovedPredictionService


class TestDependencyInjection:
    """Тестове за dependency injection функционалност"""
    
    def setup_method(self):
        """Setup за всеки тест"""
        self.client = TestClient(app)
        self.service_manager = get_service_manager()
    
    def test_service_manager_initialization(self):
        """Тест за инициализация на ServiceManager"""
        # ServiceManager трябва да съществува
        assert self.service_manager is not None
        
        # Трябва да има методи за получаване на services
        assert hasattr(self.service_manager, 'get_prediction_service')
        assert hasattr(self.service_manager, 'get_improved_prediction_service')
        assert hasattr(self.service_manager, 'get_service_status')
    
    def test_mock_prediction_service(self):
        """Тест за mock-ване на PredictionService"""
        # Създаваме mock service
        mock_service = Mock(spec=PredictionService)
        mock_service.health_check.return_value = {
            'models_loaded': True,
            'status': 'healthy'
        }
        mock_service.predict.return_value = {
            'prediction_1x2': {
                'prob_home_win': 0.4,
                'prob_draw': 0.3,
                'prob_away_win': 0.3,
                'predicted_outcome': '1',
                'confidence': 0.4
            }
        }
        
        # Заместваме service в manager
        self.service_manager.set_service('prediction', mock_service)
        
        # Тестваме че mock-ът работи
        service = self.service_manager.get_prediction_service()
        assert service == mock_service
        
        health = service.health_check()
        assert health['models_loaded'] is True
    
    def test_mock_improved_prediction_service(self):
        """Тест за mock-ване на ImprovedPredictionService"""
        # Създаваме mock service
        mock_service = Mock(spec=ImprovedPredictionService)
        mock_service.predict_with_confidence.return_value = {
            'prediction_1x2': {
                'prob_home_win': 0.5,
                'prob_draw': 0.3,
                'prob_away_win': 0.2,
                'predicted_outcome': '1',
                'confidence': 0.5
            },
            'data_quality': {
                'overall_confidence': 0.8,
                'confidence_level': 'High'
            },
            'feature_quality': {
                '1x2_model': {
                    'data_quality_score': 1.0,
                    'missing_features': [],
                    'imputed_count': 0
                }
            }
        }
        
        # Заместваме service в manager
        self.service_manager.set_service('improved_prediction', mock_service)
        
        # Тестваме че mock-ът работи
        service = self.service_manager.get_improved_prediction_service()
        assert service == mock_service
        
        result = service.predict_with_confidence(
            home_team="Test Home",
            away_team="Test Away",
            league="Test League"
        )
        assert result['data_quality']['overall_confidence'] == 0.8
    
    def test_service_isolation(self):
        """Тест за изолация между services"""
        # Създаваме различни mock services
        mock_prediction = Mock(spec=PredictionService)
        mock_improved = Mock(spec=ImprovedPredictionService)
        
        # Задаваме различни services
        self.service_manager.set_service('prediction', mock_prediction)
        self.service_manager.set_service('improved_prediction', mock_improved)
        
        # Проверяваме че са различни
        service1 = self.service_manager.get_prediction_service()
        service2 = self.service_manager.get_improved_prediction_service()
        
        assert service1 != service2
        assert service1 == mock_prediction
        assert service2 == mock_improved
    
    def test_service_status(self):
        """Тест за service status"""
        # Задаваме mock services
        mock_prediction = Mock(spec=PredictionService)
        mock_improved = Mock(spec=ImprovedPredictionService)
        
        self.service_manager.set_service('prediction', mock_prediction)
        self.service_manager.set_service('improved_prediction', mock_improved)
        
        # Получаваме status
        status = self.service_manager.get_service_status()
        
        assert status['initialized'] is True
        assert status['total_services'] == 2
        assert 'prediction' in status['services']
        assert 'improved_prediction' in status['services']
        assert status['services']['prediction']['available'] is True
        assert status['services']['improved_prediction']['available'] is True
    
    def test_service_cleanup(self):
        """Тест за cleanup на services"""
        # Задаваме mock service с cleanup метод
        mock_service = Mock(spec=PredictionService)
        mock_service.cleanup = AsyncMock()
        
        self.service_manager.set_service('prediction', mock_service)
        
        # Извикваме cleanup
        import asyncio
        asyncio.run(self.service_manager.cleanup())
        
        # Проверяваме че services са изчистени
        status = self.service_manager.get_service_status()
        assert status['initialized'] is False
        assert status['total_services'] == 0
    
    def test_hot_reload_capability(self):
        """Тест за hot-reload възможности"""
        # Първоначален service
        mock_service_v1 = Mock(spec=PredictionService)
        mock_service_v1.version = "1.0"
        
        self.service_manager.set_service('prediction', mock_service_v1)
        
        # Проверяваме първата версия
        service = self.service_manager.get_prediction_service()
        assert service.version == "1.0"
        
        # "Hot reload" с нова версия
        mock_service_v2 = Mock(spec=PredictionService)
        mock_service_v2.version = "2.0"
        
        self.service_manager.set_service('prediction', mock_service_v2)
        
        # Проверяваме новата версия
        service = self.service_manager.get_prediction_service()
        assert service.version == "2.0"
    
    def test_concurrent_access(self):
        """Тест за concurrent достъп до services"""
        import threading
        import time
        
        # Задаваме service
        mock_service = Mock(spec=PredictionService)
        mock_service.predict.return_value = {'result': 'success'}
        
        self.service_manager.set_service('prediction', mock_service)
        
        results = []
        
        def worker():
            service = self.service_manager.get_prediction_service()
            result = service.predict()
            results.append(result)
        
        # Създаваме няколко threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Чакаме всички threads
        for thread in threads:
            thread.join()
        
        # Проверяваме резултатите
        assert len(results) == 5
        for result in results:
            assert result['result'] == 'success'


class TestAPIEndpoints:
    """Тестове за API endpoints с dependency injection"""
    
    def setup_method(self):
        """Setup за всеки тест"""
        self.client = TestClient(app)
    
    def test_health_endpoint_with_mock(self):
        """Тест за health endpoint с mock service"""
        # Този тест демонстрира как можем да тестваме endpoints
        # без да зареждаме реалните модели
        
        # В реален тест бихме mock-нали dependency-то
        # За сега просто проверяваме че endpoint-ът съществува
        response = self.client.get("/health")
        
        # Ако services не са инициализирани, очакваме грешка
        # Ако са инициализирани, очакваме success
        assert response.status_code in [200, 503]
    
    def test_services_status_endpoint(self):
        """Тест за services status endpoint"""
        response = self.client.get("/services/status")
        
        # Endpoint-ът трябва да съществува
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert 'service_manager' in data
            assert 'uptime_seconds' in data
            assert 'timestamp' in data


if __name__ == "__main__":
    # Пример за ръчно тестване
    test_di = TestDependencyInjection()
    test_di.setup_method()
    
    print("🧪 Тестване на dependency injection...")
    
    try:
        test_di.test_service_manager_initialization()
        print("✅ ServiceManager initialization test passed")
        
        test_di.test_mock_prediction_service()
        print("✅ Mock PredictionService test passed")
        
        test_di.test_service_isolation()
        print("✅ Service isolation test passed")
        
        test_di.test_hot_reload_capability()
        print("✅ Hot reload test passed")
        
        print("\n🎉 Всички тестове преминаха успешно!")
        print("✨ Dependency injection архитектурата работи отлично!")
        
    except Exception as e:
        print(f"❌ Тест се провали: {e}")
        raise
