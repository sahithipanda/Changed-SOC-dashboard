from typing import List, Dict, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime, timedelta
import re
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TextClassifier:
    """Simple text classifier using TF-IDF and Naive Bayes"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.classifier = MultinomialNB()
        self.is_trained = False
        self._init_training_data()
    
    def _init_training_data(self):
        """Initialize with some basic training data"""
        self.training_texts = [
            "malware detected in system files",
            "suspicious executable downloaded",
            "ransomware encryption detected",
            "phishing email received",
            "credential theft attempt",
            "ddos attack detected",
            "high traffic volume",
            "port scanning detected"
        ]
        self.training_labels = [
            "malware",
            "malware",
            "ransomware",
            "phishing",
            "phishing",
            "ddos",
            "ddos",
            "reconnaissance"
        ]
        
        # Train initial model
        self._train_model()
    
    def _train_model(self):
        """Train the text classification model"""
        if not self.training_texts:
            return
        
        X = self.vectorizer.fit_transform(self.training_texts)
        self.classifier.fit(X, self.training_labels)
        self.is_trained = True
    
    def classify(self, text: str) -> Dict:
        """Classify text and return prediction with confidence"""
        if not self.is_trained:
            return {'label': 'unknown', 'confidence': 0.0}
        
        # Transform text
        X = self.vectorizer.transform([text])
        
        # Get prediction and probability
        label = self.classifier.predict(X)[0]
        probs = self.classifier.predict_proba(X)[0]
        confidence = max(probs)
        
        return {
            'label': label,
            'confidence': float(confidence)
        }
    
    def add_training_data(self, text: str, label: str):
        """Add new training data and retrain model"""
        self.training_texts.append(text)
        self.training_labels.append(label)
        self._train_model()

class AdvancedThreatDetector:
    """Advanced threat detection using ensemble methods"""
    def __init__(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100)
        self.threat_patterns = self._load_threat_patterns()
        self.threshold = 0.85
    
    def _load_threat_patterns(self) -> Dict:
        """Load known threat patterns and signatures"""
        return {
            'malware': [
                r'suspicious_exe_pattern',
                r'known_malware_signature',
                r'unusual_system_calls'
            ],
            'phishing': [
                r'suspicious_url_pattern',
                r'credential_theft_attempt',
                r'social_engineering_keywords'
            ],
            'ddos': [
                r'traffic_spike_pattern',
                r'connection_flood',
                r'bandwidth_exhaustion'
            ]
        }
    
    def analyze_behavior(self, data: Dict) -> Dict:
        """Analyze system behavior for threats"""
        results = {
            'anomaly_score': 0.0,
            'threat_indicators': [],
            'confidence': 0.0
        }
        
        # Analyze system calls
        if 'system_calls' in data:
            results['system_calls_score'] = self._analyze_system_calls(data['system_calls'])
        
        # Analyze network traffic
        if 'network_traffic' in data:
            results['network_score'] = self._analyze_network_traffic(data['network_traffic'])
        
        # Analyze file operations
        if 'file_operations' in data:
            results['file_ops_score'] = self._analyze_file_operations(data['file_operations'])
        
        # Calculate overall threat score
        results['threat_score'] = self._calculate_threat_score(results)
        
        return results
    
    def _analyze_system_calls(self, calls: List[str]) -> float:
        """Analyze system calls for suspicious patterns"""
        suspicious_patterns = [
            'create_remote_thread',
            'write_process_memory',
            'create_process',
            'reg_create_key'
        ]
        
        score = 0.0
        for call in calls:
            if any(pattern in call.lower() for pattern in suspicious_patterns):
                score += 0.1
        return min(score, 1.0)
    
    def _analyze_network_traffic(self, traffic: Dict) -> float:
        """Analyze network traffic for suspicious patterns"""
        score = 0.0
        
        # Check for unusual ports
        if traffic.get('unusual_ports', 0) > 5:
            score += 0.3
        
        # Check for high traffic volume
        if traffic.get('bytes_per_second', 0) > 1000000:  # 1 MB/s
            score += 0.3
        
        # Check for multiple failed connections
        if traffic.get('failed_connections', 0) > 10:
            score += 0.4
        
        return min(score, 1.0)
    
    def _analyze_file_operations(self, operations: List[Dict]) -> float:
        """Analyze file operations for suspicious activity"""
        suspicious_operations = {
            'encrypt': 0.4,
            'delete_multiple': 0.3,
            'modify_system': 0.4,
            'create_executable': 0.2
        }
        
        score = 0.0
        for op in operations:
            op_type = op.get('type', '').lower()
            if op_type in suspicious_operations:
                score += suspicious_operations[op_type]
        
        return min(score, 1.0)
    
    def _calculate_threat_score(self, results: Dict) -> float:
        """Calculate overall threat score"""
        weights = {
            'system_calls_score': 0.4,
            'network_score': 0.4,
            'file_ops_score': 0.2
        }
        
        score = 0.0
        for metric, weight in weights.items():
            if metric in results:
                score += results[metric] * weight
        
        return score

class MLAnalyzer:
    def __init__(self, config: Dict):
        self.config = config
        self._init_models()
        self.advanced_detector = AdvancedThreatDetector()
        self.threat_memory = {}
        self.correlation_window = timedelta(hours=24)
    
    def _init_models(self):
        """Initialize ML models"""
        # Initialize text classifier
        self.text_classifier = TextClassifier()
        
        # Initialize anomaly detection models
        self.isolation_forest = IsolationForest(contamination=0.1)
        self.one_class_svm = OneClassSVM(kernel='rbf', nu=0.1)
        
        # Initialize clustering model
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    def analyze_threat(self, data: Dict) -> Dict:
        """Comprehensive threat analysis"""
        results = {
            'timestamp': datetime.utcnow(),
            'basic_analysis': {},
            'advanced_analysis': {},
            'correlations': [],
            'risk_score': 0.0
        }
        
        # Basic threat detection
        if 'text' in data:
            try:
                results['basic_analysis']['classification'] = self.text_classifier.classify(data['text'])
            except Exception as e:
                print(f"Error in text classification: {e}")
                results['basic_analysis']['classification'] = {'label': 'unknown', 'confidence': 0.0}
        
        if 'features' in data:
            try:
                results['basic_analysis']['anomalies'] = self._detect_anomalies(data['features'])
                results['basic_analysis']['clusters'] = self._cluster_threats(data['features'])
            except Exception as e:
                print(f"Error in anomaly/cluster detection: {e}")
        
        # Advanced threat detection
        try:
            results['advanced_analysis'] = self.advanced_detector.analyze_behavior(data)
        except Exception as e:
            print(f"Error in advanced analysis: {e}")
        
        # Threat correlation
        try:
            results['correlations'] = self._correlate_threats(data)
        except Exception as e:
            print(f"Error in threat correlation: {e}")
        
        # Calculate final risk score
        results['risk_score'] = self._calculate_combined_risk_score(results)
        
        # Update threat memory
        self._update_threat_memory(data, results)
        
        return results
    
    def _detect_anomalies(self, features: np.ndarray) -> Dict:
        """Detect anomalies using multiple methods"""
        return {
            'isolation_forest': self.isolation_forest.fit_predict(features),
            'one_class_svm': self.one_class_svm.fit_predict(features)
        }
    
    def _cluster_threats(self, features: np.ndarray) -> np.ndarray:
        """Cluster similar threats"""
        return self.dbscan.fit_predict(features)
    
    def _correlate_threats(self, data: Dict) -> List[Dict]:
        """Correlate current threat with historical threats"""
        correlations = []
        current_time = datetime.utcnow()
        
        for threat_id, threat_data in self.threat_memory.items():
            if current_time - threat_data['timestamp'] <= self.correlation_window:
                correlation_score = self._calculate_correlation_score(data, threat_data)
                if correlation_score > 0.7:  # Correlation threshold
                    correlations.append({
                        'threat_id': threat_id,
                        'correlation_score': correlation_score,
                        'correlation_type': self._determine_correlation_type(data, threat_data)
                    })
        
        return correlations
    
    def _calculate_correlation_score(self, current: Dict, historical: Dict) -> float:
        """Calculate correlation score between two threats"""
        score = 0.0
        
        # Compare IP addresses
        if current.get('source_ip') == historical.get('source_ip'):
            score += 0.4
        
        # Compare threat types
        if current.get('type') == historical.get('type'):
            score += 0.3
        
        # Compare attack patterns
        if self._compare_patterns(current.get('pattern'), historical.get('pattern')):
            score += 0.3
        
        return score
    
    def _determine_correlation_type(self, current: Dict, historical: Dict) -> str:
        """Determine the type of correlation between threats"""
        if current.get('source_ip') == historical.get('source_ip'):
            return 'same_source'
        elif current.get('type') == historical.get('type'):
            return 'same_attack_type'
        elif self._compare_patterns(current.get('pattern'), historical.get('pattern')):
            return 'similar_pattern'
        return 'unknown'
    
    def _compare_patterns(self, pattern1: str, pattern2: str) -> bool:
        """Compare two attack patterns for similarity"""
        if not (pattern1 and pattern2):
            return False
        return pattern1.lower() == pattern2.lower()
    
    def _update_threat_memory(self, data: Dict, results: Dict):
        """Update threat memory with new threat data"""
        threat_id = f"threat_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.threat_memory[threat_id] = {
            'timestamp': datetime.utcnow(),
            'data': data,
            'analysis_results': results
        }
        
        # Clean up old threats
        self._cleanup_threat_memory()
    
    def _cleanup_threat_memory(self):
        """Remove old threats from memory"""
        current_time = datetime.utcnow()
        self.threat_memory = {
            tid: data for tid, data in self.threat_memory.items()
            if current_time - data['timestamp'] <= self.correlation_window
        }
    
    def _calculate_combined_risk_score(self, results: Dict) -> float:
        """Calculate combined risk score from all analyses"""
        weights = {
            'basic_classification': 0.3,
            'anomaly_detection': 0.2,
            'advanced_analysis': 0.3,
            'correlation': 0.2
        }
        
        score = 0.0
        
        # Basic classification score
        if 'basic_analysis' in results and 'classification' in results['basic_analysis']:
            score += weights['basic_classification'] * results['basic_analysis']['classification'].get('confidence', 0)
        
        # Anomaly detection score
        if 'basic_analysis' in results and 'anomalies' in results['basic_analysis']:
            anomaly_scores = results['basic_analysis']['anomalies']
            score += weights['anomaly_detection'] * (
                1 if -1 in anomaly_scores.get('isolation_forest', []) else 0
            )
        
        # Advanced analysis score
        if 'advanced_analysis' in results:
            score += weights['advanced_analysis'] * results['advanced_analysis'].get('threat_score', 0)
        
        # Correlation score
        if results['correlations']:
            correlation_score = max(corr['correlation_score'] for corr in results['correlations'])
            score += weights['correlation'] * correlation_score
        
        return min(score * 10, 10)  # Scale to 0-10

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length
        
    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.sequence_length],
            self.data[idx + self.sequence_length]
        )

class TimeSeriesAnalyzer:
    """Time series analysis for threat prediction using ARIMA and LSTM ensemble"""
    def __init__(self):
        self.arima_model = None
        self.lstm_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = pd.DataFrame(columns=['timestamp', 'threat_count'])
        self.forecast_steps = 48  # Extended forecast window (48 hours)
        self.last_known_timestamp = None
        self.sequence_length = 24  # Length of input sequences for LSTM
        self.min_samples = 24  # Minimum samples needed for forecasting
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def add_data_point(self, timestamp: datetime, threat_count: int):
        """Add a new data point to the time series"""
        try:
            new_data = pd.DataFrame({
                'timestamp': [timestamp],
                'threat_count': [max(1, float(threat_count))]  # Ensure minimum value of 1
            })
            
            if self.history.empty:
                self.history = new_data
            else:
                self.history = pd.concat([self.history, new_data], ignore_index=True)
            
            self.history = self.history.sort_values('timestamp').drop_duplicates(subset=['timestamp'])
            self.last_known_timestamp = timestamp
            
        except Exception as e:
            print(f"Error adding data point: {e}")
        
    def _create_lstm_model(self) -> LSTMModel:
        """Create LSTM model"""
        model = LSTMModel(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            output_size=1
        ).to(self.device)
        return model
        
    def train_model(self) -> bool:
        """Train both ARIMA and LSTM models"""
        try:
            arima_data, lstm_data = self.prepare_data()
            if arima_data is None or lstm_data is None:
                return False
                
            # Train ARIMA
            self.arima_model = ARIMA(arima_data, order=(2,1,1))
            self.arima_model = self.arima_model.fit()
            
            # Train LSTM
            if len(self.history) >= self.sequence_length:
                # Scale the data
                scaled_data = self.scaler.fit_transform(
                    self.history['threat_count'].values.reshape(-1, 1)
                )
                
                # Create sequences using numpy operations
                X = np.array([
                    scaled_data[i:(i + self.sequence_length)]
                    for i in range(len(scaled_data) - self.sequence_length)
                ])
                y = scaled_data[self.sequence_length:]
                
                # Convert to tensors
                X = torch.FloatTensor(X).to(self.device)
                y = torch.FloatTensor(y).to(self.device)
                
                # Create dataset and dataloader
                dataset = torch.utils.data.TensorDataset(X, y)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                # Initialize model and optimizer
                self.lstm_model = self._create_lstm_model()
                optimizer = torch.optim.Adam(self.lstm_model.parameters(), lr=0.001)
                criterion = nn.MSELoss()
                
                # Training loop
                self.lstm_model.train()
                for epoch in range(50):
                    total_loss = 0
                    for batch_x, batch_y in dataloader:
                        optimizer.zero_grad()
                        outputs = self.lstm_model(batch_x)
                        loss = criterion(outputs, batch_y)
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
            
            return True
            
        except Exception as e:
            print(f"Error training models: {e}")
            return False
            
    def _predict_lstm(self, steps: int) -> np.ndarray:
        """Generate LSTM predictions"""
        if self.lstm_model is None:
            return None
            
        try:
            self.lstm_model.eval()
            with torch.no_grad():
                # Get the last sequence
                scaled_data = self.scaler.transform(
                    self.history['threat_count'].values[-self.sequence_length:].reshape(-1, 1)
                )
                current_sequence = torch.FloatTensor(scaled_data).to(self.device)
                
                predictions = []
                for _ in range(steps):
                    # Predict next value
                    model_input = current_sequence.view(1, self.sequence_length, 1)
                    next_pred = self.lstm_model(model_input)
                    next_pred = next_pred.cpu().numpy()[0]
                    predictions.append(next_pred)
                    
                    # Update sequence
                    current_sequence = torch.roll(current_sequence, -1, dims=0)
                    current_sequence[-1] = torch.tensor(next_pred).to(self.device)
                
                # Inverse transform predictions
                predictions = np.array(predictions).reshape(-1, 1)
                return self.scaler.inverse_transform(predictions).flatten()
                
        except Exception as e:
            print(f"Error in LSTM prediction: {e}")
            return None
            
    def forecast(self) -> Dict:
        """Generate ensemble forecast combining ARIMA and LSTM predictions"""
        if len(self.history) < self.min_samples:
            return self._generate_simple_forecast()
            
        if self.arima_model is None or self.lstm_model is None:
            if not self.train_model():
                return self._generate_simple_forecast()
                
        try:
            # Generate base forecasts
            arima_forecast = self.arima_model.forecast(steps=self.forecast_steps)
            lstm_forecast = self._predict_lstm(self.forecast_steps)
            
            if lstm_forecast is None:
                lstm_forecast = arima_forecast
            
            # Prepare forecast times
            last_timestamp = self.last_known_timestamp or self.history['timestamp'].max()
            forecast_times = pd.date_range(
                start=last_timestamp,
                periods=self.forecast_steps + 1,
                freq='h'
            )[1:]
            
            # Combine predictions with weights (0.6 ARIMA, 0.4 LSTM)
            forecast_values = []
            for i, (ts, arima_val, lstm_val) in enumerate(zip(forecast_times, arima_forecast, lstm_forecast)):
                # Weighted combination
                base_val = 0.6 * arima_val + 0.4 * lstm_val
                
                # Add time-based patterns
                hour_factor = 1.2 + np.sin(ts.hour * np.pi / 12.0) * 0.3
                day_factor = 1.1 + np.sin(ts.hour * np.pi / 24.0) * 0.2
                
                # Add trend and noise
                trend_factor = 1.0 + (i / self.forecast_steps) * 0.15 * np.random.choice([-1, 1])
                noise = np.random.normal(1.0, 0.1 + (i / self.forecast_steps) * 0.1)
                
                # Combine all factors
                adjusted_val = base_val * hour_factor * day_factor * trend_factor * noise
                forecast_values.append(max(1, adjusted_val))
            
            return self._format_forecast_output(forecast_times, forecast_values, last_timestamp)
            
        except Exception as e:
            print(f"Forecasting error: {e}")
            return self._generate_simple_forecast()
    
    def _generate_simple_forecast(self) -> Dict:
        """Generate a simple forecast when insufficient data or errors occur"""
        last_timestamp = self.last_known_timestamp or datetime.utcnow()
        forecast_times = pd.date_range(
            start=last_timestamp,
            periods=self.forecast_steps + 1,
            freq='h'
        )[1:]
        
        # Generate simple forecasts based on time patterns
        forecast_values = []
        for i, ts in enumerate(forecast_times):
            base_val = 5.0  # Base threat level
            hour_factor = 1.2 + np.sin(ts.hour * np.pi / 12.0) * 0.3
            day_factor = 1.1 + np.sin(ts.hour * np.pi / 24.0) * 0.2
            noise = np.random.normal(1.0, 0.1)
            
            val = base_val * hour_factor * day_factor * noise
            forecast_values.append(max(1, val))
        
        return self._format_forecast_output(forecast_times, forecast_values, last_timestamp)
    
    def _format_forecast_output(self, forecast_times, forecast_values, last_timestamp) -> Dict:
        """Format the forecast output consistently"""
        # Calculate confidence intervals
        base_std = np.std(forecast_values) if len(forecast_values) > 1 else 1.0
        conf_intervals = {
            'lower': [],
            'upper': []
        }
        
        for i, val in enumerate(forecast_values):
            time_factor = 1.0 + (i / self.forecast_steps) * 0.4  # Reduced uncertainty growth
            adjusted_std = base_std * time_factor
            conf_intervals['lower'].append(max(1, val - 1.96 * adjusted_std))
            conf_intervals['upper'].append(val + 1.96 * adjusted_std)
        
        # Combine historical and forecast data
        all_data = []
        
        # Add historical data
        historical_data = self.history.copy()
        historical_data = historical_data.sort_values('timestamp')
        for _, row in historical_data.iterrows():
            if row['timestamp'] <= last_timestamp:
                all_data.append({
                    'timestamp': row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_threats': max(1, round(row['threat_count']))
                })
        
        # Add forecast data
        for ts, val in zip(forecast_times, forecast_values):
            if ts > last_timestamp:
                all_data.append({
                    'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_threats': max(1, round(val))
                })
        
        return {
            'success': True,
            'forecast': all_data,
            'confidence_intervals': {
                'lower': [max(1, round(val)) for val in conf_intervals['lower']],
                'upper': [max(1, round(val)) for val in conf_intervals['upper']]
            },
            'last_known_timestamp': last_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'model_weights': {
                'arima': 0.6,
                'lstm': 0.4
            }
        }
        
    def prepare_data(self) -> Tuple[pd.Series, np.ndarray]:
        """Prepare time series data for both ARIMA and LSTM"""
        if len(self.history) < self.min_samples:
            return None, None
            
        try:
            # Prepare base data
            data = self.history.copy()
            data['threat_count'] = pd.to_numeric(data['threat_count'], errors='coerce')
            data = data.set_index('timestamp')
            
            # Ensure proper datetime index
            data.index = pd.DatetimeIndex(data.index)
            
            # Resample and fill missing values
            data = data.resample('h').ffill()  # Use ffill() instead of fillna(method='ffill')
            data = data.interpolate(method='linear', limit=3)
            data = data.ffill()  # Use ffill() instead of fillna(method='ffill')
            data = data.fillna(1.0)  # Fill any remaining NaNs with 1.0
            
            if len(data) < self.min_samples:
                return None, None
                
            # Data for ARIMA
            arima_data = data['threat_count'].astype(float)
            
            # Data for LSTM with proper feature names
            data_for_scaling = pd.DataFrame(data['threat_count'], columns=['threat_count'])
            scaled_data = self.scaler.fit_transform(data_for_scaling)
            lstm_data = self._create_sequences(scaled_data)
            
            return arima_data, lstm_data
            
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None, None
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length), 0])
            y.append(data[i + self.sequence_length, 0])
        return np.array(X), np.array(y) 