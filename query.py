from simple_transformer.lm import LM

lm = LM('model.pt')
result = lm.query('clever, and rich, with a comfortable home and happy disposition, xyzzy to unite some of the best xyzzy of existence; and had lived nearly twenty- one xyzzy in the world with very little to distress or vex her. she was the xyzzy of the two xyzzy of a most affectionate, indulgent father; and had, in')
print(result)
