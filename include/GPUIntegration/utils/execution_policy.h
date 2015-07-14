#pragma once

namespace portable{
class ExecutionPolicy {
	public:
	enum ConfigurationState {
		Automatic = 0,
		SharedMem = 1,
		BlockSize = 2,
		GridSize = 4,
		FullManual = GridSize | BlockSize | SharedMem
	};

	ExecutionPolicy(int gridSize=0, int blockSize=0, size_t sharedMemBytes=0)
	: mState(Automatic) {
		setGridSize(gridSize);
		setBlockSize(blockSize);
		setSharedMemBytes(sharedMemBytes);  
	}
	  
	~ExecutionPolicy() {}

	int    getConfigState()    const { return mState;          }
	int    getGridSize()       const { return mGridSize;       }
	int    getBlockSize()      const { return mBlockSize;      }
	int    getMaxBlockSize()   const { return mMaxBlockSize;   }
	size_t getSharedMemBytes() const { return mSharedMemBytes; }

	void setGridSize(int arg) { 
		mGridSize = arg;  
		if (mGridSize > 0) mState |= GridSize; 
		else mState &= (FullManual - GridSize);
	}   
	void setBlockSize(int arg) {
		mBlockSize = arg; 
		if (mBlockSize > 0) mState |= BlockSize; 
		else mState &= (FullManual - BlockSize);
	}
	void setMaxBlockSize(int arg) {
		mMaxBlockSize = arg;
	}
	void setSharedMemBytes(size_t arg) { 
		mSharedMemBytes = arg; 
		mState |= SharedMem; 
	}

	private:
	int    mState;
	int    mGridSize;
	int    mBlockSize;
	int    mMaxBlockSize;
	size_t mSharedMemBytes;
};

}	// namespace portable
