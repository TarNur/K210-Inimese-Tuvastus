# person detector boot.py

import sensor, image, lcd, time
import KPU as kpu
import gc, sys

def lcd_show_except(e):
    import uio
    err_str = uio.StringIO()
    sys.print_exception(e, err_str)
    err_str = err_str.getvalue()
    img = image.Image(size=(320,224))
    img.draw_string(0, 10, err_str, scale=1, color=(0xff,0x00,0x00))
    lcd.display(img)

def formatTime(timeRan, timeUnseen):
	formattedTimeMinutes, formattedTimeSeconds = divmod(timeRan, 60)
    formattedTimeHours, formattedTimeMinutes = divmod(formattedTimeMinutes, 60)

    unseenTimeMinutes, unseenTimeSeconds = divmod(timeUnseen, 60)
    unseenTimeHours, unseenTimeMinutes = divmod(unseenTimeMinutes, 60)

    result = "%dh:%dm/%dh:%dm(unseen/time)" %(unseenTimeHours, unseenTimeMinutes, formattedTimeHours,formattedTimeMinutes)
    return result

def main(anchors, labels = None, model_addr="/sd/mobilnet7_5.kmodel", sensor_window=(224, 224), lcd_rotation=0, sensor_hmirror=False, sensor_vflip=False):
    sensor.reset()
    sensor.set_pixformat(sensor.RGB565)
    sensor.set_framesize(sensor.QVGA)
    sensor.set_windowing(sensor_window)
    sensor.set_hmirror(False)
    sensor.set_vflip(True)
    sensor.skip_frames(10)
    sensor.run(1)

    lcd.init(type=1)
    lcd.rotation(lcd_rotation)
    lcd.clear(lcd.WHITE)

    try:
        timeRan = 0
        timeUnseen = 0
        lastSeen = 0
	endTime = 0
        task = None
        task = kpu.load(model_addr)
        a = kpu.set_outputs(task, 0, 7, 7, 30)
        a = kpu.init_yolo2(task, 0.41, 0.3, 5, anchors) # threshold:[0,1], nms_value: [0, 1]
        clock = time.clock()
        while(True):
            clock.tick()
            time.sleep_ms(40)
            startTime = time.ticks_ms()/1000
            img = sensor.snapshot()
            fps = clock.fps()
            a = img.pix_to_ai()
            t = time.ticks_ms()
            objects = kpu.run_yolo2(task, img)
            t = time.ticks_ms() - t
            people = 0

            if objects:
                lastSeen = timeRan
                for obj in objects:
                    pos = obj.rect()
                    people = people + 1
                    a = img.draw_rectangle(pos)
                    a = img.draw_string(pos[0], pos[1], "%s" %(labels[obj.classid()]), scale=2, color=(255, 0, 0))

            a = img.draw_string(0, 0, "people:%d" %(people), scale=2, color=(255, 0, 0))
            a = img.draw_string(140,0, ("%2.1ffps" %(fps)), color=(255,0,0), scale=2)

            timeRan = timeRan + endTime
            if (timeRan - lastSeen) > 180:
                timeUnseen = timeUnseen + endTime

            formattedTime = formatTime(timeRan, timeUnseen)

            a = img.draw_string(0, 205, "%s" %(formattedTime), scale=1.5, color=(255, 0, 0))

            img = img.resize(320,224)

            a = lcd.display(img)
		
	    endTime = (time.ticks_ms()/1000) - startTime
    except Exception as e:
        raise e
    finally:
        if not task is None:
            a = kpu.deinit(task)


if __name__ == "__main__":
    try:
        labels = ["person"]
        anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
        #main(anchors = anchors, labels=labels, model_addr=0x180000, lcd_rotation=0)
        main(anchors = anchors, labels=labels, model_addr="/sd/mobilnet7_5.kmodel")
    except Exception as e:
        sys.print_exception(e)
        lcd_show_except(e)
    finally:
        gc.collect()
