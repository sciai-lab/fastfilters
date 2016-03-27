#ifndef FASTFILTERS_VECTOR_HXX
#define FASTFILTERS_VECTOR_HXX

// faster than std::vector because of missing out-of-bounds safety checks
template <typename T> class ConstantVector
{
  public:
    inline ConstantVector(const unsigned int size)
    {
        ptr = new T[size];
    }

    inline ~ConstantVector()
    {
        delete ptr;
    }

    inline const T &operator[](unsigned int i) const
    {
        return ptr[i];
    }

    inline T &operator[](unsigned int i)
    {
        return ptr[i];
    }

    inline T *operator+(unsigned int i)
    {
        return ptr + i;
    }

  private:
    T *ptr;
};

#endif