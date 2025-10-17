import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, ActivityIndicator } from 'react-native';
import { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api';

export default function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [apiStatus, setApiStatus] = useState('Checking...');
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    checkApiStatus();
    checkModelStatus();
  }, []);

  const checkApiStatus = async () => {
    try {
      const response = await axios.get('http://localhost:8000/health');
      setApiStatus('✅ Connected');
    } catch (error) {
      setApiStatus('❌ Offline');
    }
  };

  const checkModelStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/rl/model/info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Model status error:', error);
    }
  };

  const initializeModel = async () => {
    setIsLoading(true);
    try {
      const response = await axios.post(`${API_BASE_URL}/rl/initialize`);
      alert('Model initialized successfully! ' + response.data.message);
      checkModelStatus();
    } catch (error) {
      alert('Error initializing model: ' + error.message);
    }
    setIsLoading(false);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>🎓 reinFORCING the people</Text>
      <Text style={styles.subtitle}>RL Language Learning</Text>
      
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>Backend: {apiStatus}</Text>
        {modelInfo && (
          <>
            <Text style={styles.statusText}>
              Model: {modelInfo.status === 'loaded' ? '✅ Loaded' : '⚠️ Not Loaded'}
            </Text>
            {modelInfo.status === 'loaded' && (
              <>
                <Text style={styles.statusText}>Epsilon: {modelInfo.epsilon.toFixed(4)}</Text>
                <Text style={styles.statusText}>Episodes: {modelInfo.training_episodes}</Text>
              </>
            )}
          </>
        )}
      </View>

      {isLoading ? (
        <ActivityIndicator size="large" color="#667eea" />
      ) : (
        <>
          {modelInfo?.status !== 'loaded' && (
            <Button
              title="🚀 Initialize RL Model"
              onPress={initializeModel}
              color="#667eea"
            />
          )}
          
          <View style={styles.buttonContainer}>
            <Button
              title="📊 Refresh Status"
              onPress={() => {
                checkApiStatus();
                checkModelStatus();
              }}
              color="#764ba2"
            />
          </View>
        </>
      )}

      <View style={styles.footer}>
        <Text style={styles.footerText}>
          🧠 Powered by Deep Q-Network
        </Text>
        <Text style={styles.footerText}>
          React Native + FastAPI + TensorFlow
        </Text>
      </View>

      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  title: {
    fontSize: 32,
    fontWeight: 'bold',
    marginBottom: 10,
    color: '#667eea',
  },
  subtitle: {
    fontSize: 18,
    color: '#764ba2',
    marginBottom: 40,
  },
  statusContainer: {
    backgroundColor: 'white',
    padding: 20,
    borderRadius: 10,
    marginBottom: 30,
    width: '100%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  statusText: {
    fontSize: 16,
    marginVertical: 5,
    color: '#333',
  },
  buttonContainer: {
    marginTop: 20,
    width: '100%',
  },
  footer: {
    position: 'absolute',
    bottom: 30,
    alignItems: 'center',
  },
  footerText: {
    fontSize: 12,
    color: '#999',
    marginVertical: 2,
  },
});
