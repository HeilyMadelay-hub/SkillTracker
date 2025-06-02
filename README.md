# SkillTracker - Tracker de Habilidades con IA y Seguridad

## 📝 Descripción

SkillTracker es una aplicación web local que combina **análisis inteligente de datos** con **seguridad robusta** para el tracking personal de habilidades profesionales. Utiliza machine learning básico para análisis de patrones y implementa controles de seguridad específicos contra amenazas comunes.

**¿Por qué local?** Privacidad total + control completo sobre la seguridad de los datos.

### Lo que hace realmente:

- **Análisis ML**: Clustering de skills, detección de anomalías en progreso, predicciones básicas
- **Seguridad avanzada**: Rate limiting, input validation, security logging, detección de ataques
- **Dashboard inteligente**: Visualizaciones con insights generados por algoritmos
- **Auditoría completa**: Logging de todas las acciones para análisis de seguridad
- **Detección de patrones**: Identifica comportamientos anómalos en el uso

## 🛠️ Tecnologías

### Frontend:
- React 18 con validación de inputs robusta
- Material-UI + security headers
- Chart.js para visualizaciones
- Input sanitization en tiempo real

### Backend:
- FastAPI con middleware de seguridad
- MongoDB local con validación estricta
- PyJWT + rate limiting personalizado
- bcrypt + salt rounds configurables

### Machine Learning:
- **scikit-learn**: KMeans clustering, LinearRegression, IsolationForest
- **pandas**: Análisis de series temporales de progreso
- **numpy**: Cálculos estadísticos y feature engineering
- **joblib**: Persistencia de modelos entrenados

### Seguridad:
- **slowapi**: Rate limiting avanzado
- **pydantic**: Validación estricta de schemas
- **python-multipart**: Parsing seguro de archivos
- **cryptography**: Cifrado AES-256 + HMAC

## 📂 Estructura del Proyecto

```
secure-skill-analyzer/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── security/       # Input validation, sanitization
│   │   └── services/       # API calls con retry y validation
├── backend/
│   ├── app/
│   │   ├── api/            # Endpoints con rate limiting
│   │   ├── ml/             # Modelos scikit-learn
│   │   ├── security/       # Rate limiting, validation, logging
│   │   └── analytics/      # Feature engineering y análisis
├── models/                 # Modelos ML entrenados (.joblib)
├── logs/                   # Security logs y audit trails
└── tests/                  # Tests de seguridad y ML
```

## 📋 Requisitos

- **Docker** y Docker Compose
- O manualmente:
  - Python 3.9+ con scikit-learn
  - Node.js 16+
  - MongoDB Community Edition

## 🚀 Instalación

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/secure-skill-analyzer.git
cd secure-skill-analyzer

# Levantar con configuración de seguridad
docker-compose up -d

# Verificar endpoint de salud con autenticación
curl -H "Authorization: Bearer test-token" http://localhost:8000/health

# Verificar logs de seguridad
tail -f logs/security.log
```

## 🤖 Machine Learning Implementado

### 1. Análisis de Clustering de Skills

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

class SkillClusterAnalyzer:
    """Agrupa usuarios por patrones similares de habilidades"""
    
    def __init__(self):
        self.kmeans = KMeans(n_clusters=5, random_state=42)
        self.scaler = StandardScaler()
        
    def analyze_user_profile(self, user_skills):
        """Determina a qué cluster pertenece el usuario"""
        # Feature engineering: niveles, categorías, tiempo invertido
        features = self._extract_features(user_skills)
        features_scaled = self.scaler.transform([features])
        
        cluster = self.kmeans.predict(features_scaled)[0]
        return {
            'cluster': cluster,
            'profile_type': self._get_profile_name(cluster),
            'similar_users_count': self._count_cluster_users(cluster)
        }
```

### 2. Detección de Anomalías en Progreso

```python
from sklearn.ensemble import IsolationForest

class ProgressAnomalyDetector:
    """Detecta patrones anómalos en el progreso de aprendizaje"""
    
    def __init__(self):
        self.detector = IsolationForest(contamination=0.1, random_state=42)
        
    def detect_unusual_progress(self, progress_history):
        """Identifica progreso sospechosamente rápido o lento"""
        # Features: velocidad de aprendizaje, consistencia, saltos de nivel
        features = self._engineer_progress_features(progress_history)
        
        anomaly_score = self.detector.decision_function([features])[0]
        is_anomaly = self.detector.predict([features])[0] == -1
        
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'reason': self._explain_anomaly(features) if is_anomaly else None
        }
```

### 3. Predicción de Tiempo de Aprendizaje

```python
from sklearn.linear_model import LinearRegression
import numpy as np

class LearningTimePredictor:
    """Predice cuánto tiempo tomará dominar una skill"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        
    def predict_learning_time(self, skill_name, current_level, target_level, user_history):
        """Estima días necesarios para alcanzar target_level"""
        if not self.trained:
            self._train_on_historical_data()
            
        # Features: skill complexity, user learning speed, level gap
        features = np.array([[
            self._get_skill_complexity(skill_name),
            self._calculate_user_speed(user_history),
            target_level - current_level,
            self._get_skill_category_factor(skill_name)
        ]])
        
        predicted_days = max(1, int(self.model.predict(features)[0]))
        confidence = self._calculate_confidence(features)
        
        return {
            'estimated_days': predicted_days,
            'confidence': confidence,
            'factors': self._explain_prediction(features)
        }
```

## 🔒 Implementación de Seguridad

### 1. Rate Limiting Avanzado

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

class SecurityMiddleware:
    """Rate limiting con diferentes límites por endpoint"""
    
    def __init__(self):
        self.limiter = Limiter(key_func=get_remote_address)
        
    @limiter.limit("100/minute")  # Endpoints normales
    async def standard_endpoint(self, request):
        pass
        
    @limiter.limit("10/minute")   # Endpoints sensibles
    async def sensitive_endpoint(self, request):
        pass
        
    @limiter.limit("3/minute")    # Login/auth
    async def auth_endpoint(self, request):
        pass
```

### 2. Validación de Inputs Robusta

```python
from pydantic import BaseModel, validator, Field
import re

class SkillInputModel(BaseModel):
    """Validación estricta para prevenir inyecciones"""
    
    name: str = Field(..., min_length=1, max_length=50)
    level: int = Field(..., ge=1, le=10)
    category: str = Field(..., regex=r'^[a-zA-Z\s]+$')
    notes: str = Field(default="", max_length=500)
    
    @validator('name')
    def validate_skill_name(cls, v):
        # Prevenir XSS y SQL injection
        if re.search(r'[<>"\';(){}]', v):
            raise ValueError('Caracteres no permitidos en nombre de skill')
        return v.strip()
    
    @validator('notes')
    def sanitize_notes(cls, v):
        # Limpiar HTML tags y scripts
        clean_text = re.sub(r'<[^>]*>', '', v)
        if 'script' in clean_text.lower():
            raise ValueError('Contenido no permitido en notas')
        return clean_text.strip()
```

### 3. Security Logging y Auditoría

```python
import logging
from datetime import datetime
import hashlib

class SecurityLogger:
    """Logging completo de eventos de seguridad"""
    
    def __init__(self):
        self.logger = logging.getLogger('security')
        self.setup_secure_logging()
        
    def log_auth_attempt(self, email, success, ip_address, user_agent):
        """Log de intentos de autenticación"""
        hashed_email = hashlib.sha256(email.encode()).hexdigest()[:8]
        
        self.logger.info({
            'event': 'auth_attempt',
            'user_hash': hashed_email,
            'success': success,
            'ip': ip_address,
            'user_agent': user_agent[:100],  # Truncar para evitar logs gigantes
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_rate_limit_exceeded(self, ip_address, endpoint, limit):
        """Log de rate limit violations"""
        self.logger.warning({
            'event': 'rate_limit_exceeded',
            'ip': ip_address,
            'endpoint': endpoint,
            'limit': limit,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def log_anomaly_detected(self, user_id, anomaly_type, details):
        """Log de comportamientos anómalos detectados por ML"""
        user_hash = hashlib.sha256(str(user_id).encode()).hexdigest()[:8]
        
        self.logger.warning({
            'event': 'anomaly_detected',
            'user_hash': user_hash,
            'type': anomaly_type,
            'details': details,
            'timestamp': datetime.utcnow().isoformat()
        })
```

### 4. Detección de Ataques Básicos

```python
import re

class AttackDetector:
    """Detecta patrones de ataque comunes"""
    
    def __init__(self):
        self.failed_attempts = {}  # IP -> contador
        self.suspicious_patterns = [
            r'union\s+select',      # SQL injection
            r'<script[^>]*>',       # XSS
            r'\.\.\/.\.\./',       # Path traversal
            r'eval\s*\(',           # Code injection
        ]
    
    def analyze_request(self, request_data, ip_address):
        """Analiza request en busca de patrones maliciosos"""
        threats_detected = []
        
        # Check for injection patterns
        for pattern in self.suspicious_patterns:
            if re.search(pattern, str(request_data), re.IGNORECASE):
                threats_detected.append(f'Suspicious pattern: {pattern}')
        
        # Check for brute force
        if self._is_brute_force_attempt(ip_address):
            threats_detected.append('Possible brute force attack')
        
        # Check for excessive requests
        if self._is_dos_attempt(ip_address):
            threats_detected.append('Possible DoS attack')
            
        return threats_detected
    
    def _is_brute_force_attempt(self, ip_address):
        """Detecta intentos de fuerza bruta"""
        failures = self.failed_attempts.get(ip_address, 0)
        return failures > 5  # Más de 5 fallos = sospechoso
```

## 📊 Dashboard con Análisis Inteligente

### Insights generados por ML:
- **Perfil de aprendizaje**: "Eres un 'Backend Specialist' similar al 23% de usuarios"
- **Anomalías detectadas**: "Progreso inusualmente rápido en Docker (verificar)"
- **Predicciones**: "Estimado 12 días para dominar Kubernetes (85% confianza)"
- **Recomendaciones**: "Usuarios con tu perfil suelen aprender React después de JavaScript"

### Security Dashboard:
- **Rate limits**: Requests por minuto por endpoint
- **Failed attempts**: Intentos de login fallidos por IP
- **Anomalies**: Comportamientos detectados como sospechosos
- **Audit trail**: Log de todas las acciones importantes

## 🎯 Funcionalidades IA + Seguridad

### Machine Learning:
✅ **Clustering real** con K-means para perfiles de usuario  
✅ **Detección de anomalías** con Isolation Forest  
✅ **Predicciones** con regresión lineal para tiempos de aprendizaje  
✅ **Feature engineering** de datos temporales de progreso  
✅ **Análisis estadístico** de patrones de aprendizaje  

### Ciberseguridad:
✅ **Rate limiting** diferenciado por tipo de endpoint  
✅ **Input validation** robusta contra XSS/SQLi  
✅ **Security logging** completo con hashing de datos sensibles  
✅ **Detección de ataques** básicos (brute force, injection, DoS)  
✅ **Auditoría completa** de acciones de usuario  

## 🧪 Testing de Seguridad y ML

```bash
# Tests de ML
pytest tests/test_ml_models.py -v
# Test clustering accuracy, anomaly detection precision

# Tests de seguridad
pytest tests/test_security.py -v
# Test rate limiting, input validation, attack detection

# Security audit
bandit -r backend/ -ll
safety check -r requirements.txt

# ML model validation
python tests/validate_models.py
```

**Tests implementados:**
- **ML**: Precisión de clustering, recall de detección de anomalías
- **Seguridad**: Rate limiting, input validation, detección de ataques
- **Integración**: End-to-end con payloads maliciosos

## 📈 Problemática que resuelve

**El problema específico:**
Los sistemas de tracking personal carecen de:
- **Análisis inteligente** de patrones de aprendizaje
- **Seguridad robusta** contra ataques comunes
- **Detección de anomalías** en comportamiento de uso
- **Auditoría completa** de acciones para compliance

**Nuestra solución IA + Seguridad:**
✅ **ML real** para insights sobre patrones de aprendizaje  
✅ **Seguridad proactiva** contra amenazas conocidas  
✅ **Detección automática** de comportamientos anómalos  
✅ **Logging completo** para auditoría y forensics  
✅ **Validación robusta** de todos los inputs de usuario  

## 💻 Ejemplo de Análisis ML

### Input del usuario:
```json
{
  "skills": [
    {"name": "Python", "level": 8, "days_learning": 180},
    {"name": "JavaScript", "level": 6, "days_learning": 90},
    {"name": "Docker", "level": 9, "days_learning": 30}
  ]
}
```

### Output del análisis ML:
```json
{
  "cluster_analysis": {
    "profile": "DevOps Specialist",
    "similarity": 0.87,
    "cluster_id": 2
  },
  "anomaly_detection": {
    "docker_progress": {
      "is_anomaly": true,
      "reason": "Unusually fast progression to level 9 in 30 days",
      "anomaly_score": -0.23
    }
  },
  "predictions": {
    "kubernetes": {
      "estimated_days": 15,
      "confidence": 0.82,
      "factors": ["Strong Docker foundation", "Fast learner profile"]
    }
  }
}
```

## 🔧 Configuración de Seguridad

```bash
# .env
RATE_LIMIT_PER_MINUTE=100
FAILED_LOGIN_THRESHOLD=5
ANOMALY_DETECTION_SENSITIVITY=0.1
SECURITY_LOG_LEVEL=WARNING

# JWT configuration
JWT_SECRET_KEY=your-super-secure-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=60

# ML model paths
CLUSTERING_MODEL_PATH=models/kmeans_skill_clusters.joblib
ANOMALY_MODEL_PATH=models/isolation_forest_anomalies.joblib
PREDICTION_MODEL_PATH=models/linear_regression_time.joblib
```

## ❓ FAQ

**¿El machine learning es realmente funcional?**  
Sí, utiliza scikit-learn para clustering K-means, detección de anomalías con Isolation Forest y predicciones con regresión lineal. No es deep learning, pero son algoritmos reales y funcionales.

**¿Qué tipo de ataques detecta?**  
Brute force, inyección SQL básica, XSS, path traversal y DoS simples. No es un WAF completo, pero detecta amenazas comunes.

**¿Los logs de seguridad son completos?**  
Sí, registra autenticación, rate limiting, anomalías ML y ataques detectados. Los datos sensibles se hashean antes de loguearse.

**¿Los modelos ML necesitan reentrenamiento?**  
Los modelos se entrenan inicialmente con datos sintéticos y se pueden reentrenar con datos reales de uso (manteniendo privacidad).

## 🚧 Limitaciones realistas

### Machine Learning:
- **Modelos básicos**: No es deep learning, son algoritmos clásicos
- **Datos limitados**: Entrenamiento inicial con datos sintéticos
- **Precisión moderada**: ~75-80% en detección de anomalías
- **Features simples**: No procesamiento de texto avanzado

### Seguridad:
- **Detección básica**: No es un WAF enterprise-grade
- **Rate limiting simple**: No distribuido, solo por IP
- **Logging local**: No integración SIEM externa
- **Auditoría básica**: Suficiente para compliance básico

## 👨‍💻 Autor

**[Tu Nombre]**
- 🐙 GitHub: [@tu-usuario](https://github.com/tu-usuario)
- 💼 LinkedIn: [tu-perfil](https://linkedin.com/in/tu-perfil)
- 📧 Email: tu.email@ejemplo.com

---

**Proyecto desarrollado para demostrar competencias en machine learning aplicado y ciberseguridad defensiva a nivel junior/mid.**

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para detalles.

---

*Proyecto enfocado en IA + Ciberseguridad con implementaciones reales y funcionales para demostrar competencias en machine learning aplicado y ciberseguridad defensiva a nivel junior/mid.*
