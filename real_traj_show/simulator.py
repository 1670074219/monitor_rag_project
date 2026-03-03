import asyncio
import httpx
import math
import random

API_URL = "http://127.0.0.1:8000/api/tracking/push"

async def simulate_track(track_id, center_x, center_y, radius, speed, duration_sec):
    async with httpx.AsyncClient() as client:
        tick = random.uniform(0, 100)
        end_time = asyncio.get_event_loop().time() + duration_sec
        
        while asyncio.get_event_loop().time() < end_time:
            # Generate moving coordinates forming a circle/ellipse over time
            x = center_x + radius * math.cos(tick * speed)
            y = center_y + radius * math.sin(tick * speed)
            
            payload = {
                "track_id": track_id,
                "x": x,
                "y": y
            }
            
            try:
                await client.post(API_URL, json=payload)
                # print(f"Pushed track {track_id}: ({x:.1f}, {y:.1f})")
            except Exception as e:
                print(f"Error pushing track {track_id}: {e}")
                
            tick += 1
            await asyncio.sleep(0.1) # 10 FPS
            
        print(f"Track {track_id} left the scene.")

async def main():
    print("Starting simulated tracker bot...")
    
    # 5 targets walking around different spots
    tasks = [
        simulate_track(1, center_x=300, center_y=300, radius=100, speed=0.08, duration_sec=30),
        simulate_track(2, center_x=600, center_y=200, radius=80, speed=0.12, duration_sec=40),
        simulate_track(3, center_x=400, center_y=600, radius=200, speed=0.05, duration_sec=25),
        simulate_track(4, center_x=800, center_y=500, radius=150, speed=0.10, duration_sec=35),
        simulate_track(5, center_x=200, center_y=700, radius=60, speed=0.15, duration_sec=20)
    ]
    
    await asyncio.gather(*tasks)
    print("Done simulation.")

if __name__ == "__main__":
    asyncio.run(main())
