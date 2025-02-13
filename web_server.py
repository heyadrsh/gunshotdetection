import asyncio
import websockets
import json
import numpy as np
from combined_gunshot_gui import AudioProcessor
import sounddevice as sd
import threading
from queue import Queue
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WebGunShotDetector:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.audio_queue = Queue()
        self.clients = set()
        self.is_processing = False
        
        # Create logs directory
        self.logs_dir = "web_detection_logs"
        os.makedirs(self.logs_dir, exist_ok=True)
        self.log_filename = os.path.join(
            self.logs_dir,
            f"web_detection_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    async def register(self, websocket):
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(f"New client connected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    async def unregister(self, websocket):
        self.clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
        with open(self.log_filename, 'a', encoding='utf-8') as f:
            f.write(f"Client disconnected at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def process_audio_data(self, audio_data):
        """Process audio data and check for gunshots."""
        try:
            # Convert audio data to numpy array
            audio_array = np.array(audio_data, dtype=np.float32)
            
            # Process audio using YAMNet
            scores, embeddings, spectrogram = self.audio_processor.yamnet_model(audio_array)
            class_scores = scores.numpy().mean(axis=0)
            
            # Check for gunshots
            detections = self.audio_processor.check_for_gunshot(
                self.audio_processor.class_names,
                class_scores
            )
            
            # Get current sounds
            top_classes = np.argsort(class_scores)[-5:][::-1]
            current_sounds = []
            for idx in top_classes:
                confidence = class_scores[idx] * 100
                if confidence > 5.0:
                    current_sounds.append(
                        f"{self.audio_processor.class_names[idx]}: {confidence:.1f}%"
                    )
            
            # If gunshot detected, classify the type
            if detections:
                gun_type, confidence, all_probs = self.audio_processor.classify_gun_type(audio_array)
                
                # Log detection
                with open(self.log_filename, 'a', encoding='utf-8') as f:
                    f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Gunshot Detected\n")
                    f.write(f"Detections: {detections}\n")
                    f.write(f"Gun Type: {gun_type} (Confidence: {confidence:.1f}%)\n")
                
                return {
                    'gunshot_detected': True,
                    'detections': detections,
                    'gun_type': gun_type,
                    'confidence': confidence,
                    'all_probabilities': [
                        {'name': self.audio_processor.label_encoder_classes[i],
                         'probability': float(prob)}
                        for i, prob in enumerate(all_probs)
                    ],
                    'current_sounds': current_sounds
                }
            
            return {
                'gunshot_detected': False,
                'current_sounds': current_sounds
            }
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            return None

    async def broadcast_results(self, results):
        """Broadcast detection results to all connected clients."""
        if not results:
            return
            
        if results['gunshot_detected']:
            # Send gunshot detection
            gunshot_message = {
                'type': 'gunshot_detected',
                'detections': results['detections']
            }
            await self.broadcast_message(gunshot_message)
            
            # Send gun type classification
            gun_type_message = {
                'type': 'gun_type',
                'gun_type': results['gun_type'],
                'confidence': results['confidence'],
                'all_probabilities': results['all_probabilities']
            }
            await self.broadcast_message(gun_type_message)
        
        # Send current sounds
        sounds_message = {
            'type': 'current_sounds',
            'sounds': results['current_sounds']
        }
        await self.broadcast_message(sounds_message)

    async def broadcast_message(self, message):
        """Send a message to all connected clients."""
        if not self.clients:
            return
            
        disconnected_clients = set()
        
        for client in self.clients:
            try:
                await client.send(json.dumps(message))
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister(client)

    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections and messages."""
        await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data['type'] == 'audio_data':
                        results = self.process_audio_data(data['data'])
                        await self.broadcast_results(results)
                    elif data['type'] == 'stop':
                        logger.info("Received stop command")
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)

async def main():
    detector = WebGunShotDetector()
    server = await websockets.serve(
        detector.handle_websocket,
        "localhost",
        8000
    )
    
    logger.info("WebSocket server started on ws://localhost:8000")
    
    try:
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
        server.close()

if __name__ == "__main__":
    asyncio.run(main()) 