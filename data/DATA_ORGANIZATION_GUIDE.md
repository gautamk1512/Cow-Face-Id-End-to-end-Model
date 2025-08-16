# Real Data Organization Guide

## 📁 Directory Structure

### Raw Data Collection
- `data/raw/humans/` - Put all raw human photos here
- `data/raw/cows/` - Put all raw cow photos here

### Processed Training Data  
- `data/train/human/` - Human face crops for training
- `data/train/cow_001/` - Cow 1 face crops for training
- `data/train/cow_002/` - Cow 2 face crops for training
- `data/train/cow_003/` - Cow 3 face crops for training

### Validation Data
- `data/val/human/` - Human face crops for validation  
- `data/val/cow_001/` - Cow 1 face crops for validation
- `data/val/cow_002/` - Cow 2 face crops for validation
- `data/val/cow_003/` - Cow 3 face crops for validation

## 🎯 Data Collection Guidelines

### For Perfect Human vs Cow Distinction:

#### Human Face Requirements:
- 📸 **Minimum 100+ images per person**
- 👤 **Multiple angles** (front, profile, 3/4 view)  
- 💡 **Different lighting** (indoor, outdoor, bright, dim)
- 🎭 **Various expressions** (neutral, smiling, serious)
- 👕 **Different clothing/backgrounds**
- 📏 **High resolution** (min 224x224 after cropping)

#### Cow Face Requirements:
- 📸 **Minimum 100+ images per cow**
- 🐄 **Multiple angles** (front, profile, 3/4 view)
- 💡 **Different lighting** (barn, field, sunny, cloudy)
- 📐 **Clear face visibility** (no obstructions)  
- 🏷️ **Consistent cow identity** (same cow in folder)
- 📏 **High resolution** (min 224x224 after cropping)

## 🔧 Processing Steps:

1. **Collect Raw Images**: Put images in raw/ folders
2. **Run Face Detection**: `python prepare_real_data.py --detect`
3. **Manual Review**: Check detected faces for quality
4. **Train Enhanced Model**: `python train_enhanced_model.py`
5. **Test Perfect Accuracy**: `python test_real_accuracy.py`

## 📊 Minimum Data Requirements for Production:

- **Humans**: 500-1000+ face crops (multiple people)
- **Cows**: 500-1000+ face crops (3-5 different cows)
- **80/20 train/val split** automatically applied
- **Balanced classes** for optimal training

## 🎉 Expected Results:

With proper real data:
- **>98% accuracy** on human vs cow distinction
- **Perfect production performance**
- **No confusion between species**
