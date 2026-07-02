from PIL import Image
import os

# Открываем повреждённый GIF
try:
    img = Image.open('schelling_tolerance_0.30.gif')

    # Сохраняем как обычный GIF с оптимизацией
    img.save('schelling_tolerance_0.30_fixed.gif',
             save_all=True,
             optimize=True,
             loop=0)

    print("✅ GIF восстановлен как schelling_tolerance_0.30_fixed.gif")
    print(f"   Размер: {os.path.getsize('schelling_tolerance_0.30_fixed.gif') / 1024:.1f} КБ")

except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("Попробуй запустить код сохранения заново")