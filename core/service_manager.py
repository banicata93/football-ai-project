"""
Service Manager - Централизирано управление на services
"""

import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import logging

from api.prediction_service import PredictionService
from api.improved_prediction_service import ImprovedPredictionService
from .utils import setup_logging


class ServiceManager:
    """
    Централизиран мениджър за всички services
    
    Предимства:
    - Единно място за инициализация
    - Thread-safe достъп
    - Лесно тестване с mock services
    - Hot-reload възможности
    - Graceful shutdown
    """
    
    def __init__(self):
        """Инициализация на ServiceManager"""
        self.logger = setup_logging()
        self._services: Dict[str, Any] = {}
        self._initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """
        Инициализира всички services
        
        Raises:
            Exception: Ако инициализацията се провали
        """
        async with self._lock:
            if self._initialized:
                self.logger.warning("Services вече са инициализирани")
                return
            
            self.logger.info("🚀 Инициализиране на services...")
            
            try:
                # Инициализация на PredictionService
                self.logger.info("Зареждане на PredictionService...")
                prediction_service = PredictionService()
                self._services['prediction'] = prediction_service
                self.logger.info("✓ PredictionService зареден")
                
                # Инициализация на ImprovedPredictionService
                self.logger.info("Зареждане на ImprovedPredictionService...")
                improved_service = ImprovedPredictionService()
                self._services['improved_prediction'] = improved_service
                self.logger.info("✓ ImprovedPredictionService зареден")
                
                self._initialized = True
                self.logger.info("🎉 Всички services инициализирани успешно")
                
            except Exception as e:
                self.logger.error(f"❌ Грешка при инициализация на services: {e}")
                await self.cleanup()
                raise
    
    async def cleanup(self) -> None:
        """
        Почиства всички services при shutdown
        """
        async with self._lock:
            if not self._initialized:
                return
            
            self.logger.info("🧹 Почистване на services...")
            
            # Cleanup на services ако имат cleanup методи
            for service_name, service in self._services.items():
                try:
                    if hasattr(service, 'cleanup'):
                        await service.cleanup()
                        self.logger.info(f"✓ {service_name} почистен")
                except Exception as e:
                    self.logger.error(f"❌ Грешка при почистване на {service_name}: {e}")
            
            self._services.clear()
            self._initialized = False
            self.logger.info("✓ Services почистени")
    
    def get_prediction_service(self) -> PredictionService:
        """
        Получава PredictionService инстанция
        
        Returns:
            PredictionService инстанция
            
        Raises:
            RuntimeError: Ако services не са инициализирани
        """
        if not self._initialized:
            raise RuntimeError("Services не са инициализирани. Извикайте initialize() първо.")
        
        service = self._services.get('prediction')
        if service is None:
            raise RuntimeError("PredictionService не е наличен")
        
        return service
    
    def get_improved_prediction_service(self) -> ImprovedPredictionService:
        """
        Получава ImprovedPredictionService инстанция
        
        Returns:
            ImprovedPredictionService инстанция
            
        Raises:
            RuntimeError: Ако services не са инициализирани
        """
        if not self._initialized:
            raise RuntimeError("Services не са инициализирани. Извикайте initialize() първо.")
        
        service = self._services.get('improved_prediction')
        if service is None:
            raise RuntimeError("ImprovedPredictionService не е наличен")
        
        return service
    
    def is_initialized(self) -> bool:
        """
        Проверява дали services са инициализирани
        
        Returns:
            True ако са инициализирани
        """
        return self._initialized
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Получава статус на всички services
        
        Returns:
            Dictionary със статус информация
        """
        return {
            'initialized': self._initialized,
            'services': {
                name: {
                    'available': service is not None,
                    'type': type(service).__name__
                }
                for name, service in self._services.items()
            },
            'total_services': len(self._services)
        }
    
    async def reinitialize(self) -> None:
        """
        Реинициализира всички services (за hot-reload)
        """
        self.logger.info("🔄 Реинициализиране на services...")
        await self.cleanup()
        await self.initialize()
        self.logger.info("✓ Services реинициализирани")
    
    def set_service(self, name: str, service: Any) -> None:
        """
        Задава service (за тестване)
        
        Args:
            name: Име на service
            service: Service инстанция
        """
        self._services[name] = service
        if not self._initialized:
            self._initialized = True
    
    def remove_service(self, name: str) -> None:
        """
        Премахва service (за тестване)
        
        Args:
            name: Име на service
        """
        if name in self._services:
            del self._services[name]


# Глобална инстанция на ServiceManager
_service_manager: Optional[ServiceManager] = None


def get_service_manager() -> ServiceManager:
    """
    Получава глобалната ServiceManager инстанция
    
    Returns:
        ServiceManager инстанция
    """
    global _service_manager
    if _service_manager is None:
        _service_manager = ServiceManager()
    return _service_manager


@asynccontextmanager
async def lifespan_context():
    """
    Context manager за FastAPI lifespan
    
    Използва се за инициализация и cleanup на services
    """
    service_manager = get_service_manager()
    
    try:
        # Startup
        await service_manager.initialize()
        yield service_manager
    finally:
        # Shutdown
        await service_manager.cleanup()


# Dependency injection функции
def get_prediction_service() -> PredictionService:
    """
    FastAPI dependency за PredictionService
    
    Returns:
        PredictionService инстанция
    """
    return get_service_manager().get_prediction_service()


def get_improved_prediction_service() -> ImprovedPredictionService:
    """
    FastAPI dependency за ImprovedPredictionService
    
    Returns:
        ImprovedPredictionService инстанция
    """
    return get_service_manager().get_improved_prediction_service()


def get_service_status() -> Dict[str, Any]:
    """
    FastAPI dependency за service status
    
    Returns:
        Service status информация
    """
    return get_service_manager().get_service_status()
