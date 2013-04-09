#ifndef CONTAINERS_H
#define CONTAINERS_H

#include <algorithm>
#include <deque>
#include <map>
#include <vector>

#ifndef NO_CXX11_STL
#include <unordered_map>
#endif

template <typename T>
class Vector : public std::vector<T>
{
private:
    typedef std::vector<T> BaseType;

public:
    Vector(): BaseType() {}
    explicit Vector(int size): BaseType((size_t)size) {}

    // Vector(int size, const T &value)
    // Vector(const Vector<T> &other)
    // ~Vector()

    void append(const T &value) { BaseType::push_back(value); }
    //T &at(int i) { return BaseType::at((typename size_type)i); }
    //T &at(size_t i);
    //const T &at(int i) const { return BaseType::at((typename size_type)i); }
    //const T &at(size_t i) const;

    // T &back()
    // const T &back() const
    // iterator begin()
    // const_iterator begin() const
    // int capacity() const
    // void clear()

    typename BaseType::const_iterator constBegin() const { return BaseType::begin(); }

    ////
    const T *constData() const { return (const T *)(&BaseType::front()); }

    typename BaseType::const_iterator constEnd() const { return BaseType::end(); }

    bool contains(const T &value) const
    {
        for (size_t i = 0; i < BaseType::size(); i++)
            if (BaseType::at(i) == value)
                return true;
        return false;
    }

    int count(const T &value) const
    {
        //int c = 0;
        //for (size_t i = 0; i < BaseType::size(); i++)
        //    if (BaseType::at(i) == value)
        //        c++;
        //return c;
        return (int)std::count(BaseType::begin(), BaseType::end(), value);
    }

    //int count() const { return size(); }
    size_t count() const { return BaseType::size(); }

    ////
    T *data() { return &BaseType::front(); }

    const T *data() const { return (const T *)(&BaseType::front()); }

    // bool empty() const
    // iterator end()
    // const_iterator end() const

    bool endsWith(const T &value) const { return (!BaseType::empty() && BaseType::back() == value); }

    // iterator erase(iterator pos)
    // iterator erase(iterator begin, iterator end)

    Vector<T> &fill(const T &value, int size = -1)
    {
        if (size >= 0)
            BaseType::resize(size);
        //for (size_t i = 0; i < BaseType::size(); i++)
        //    BaseType::at(i) = value;
        std::fill(BaseType::begin(), BaseType::end(), value);
        return *this;
    }

    T &first() { return BaseType::front(); }
    const T &first() const { return BaseType::front(); }

    // T &front()
    // const T &front() const

    int indexOf(const T &value, int from = 0) const
    {
        if (from < 0)
            from = std::max(from + (int)BaseType::size(), 0);
        for (size_t i = (size_t)from; i < BaseType::size(); i++)
            if (BaseType::at(i) == value)
                return (int)i;
        return -1;
    }

    void insert(int i, const T &value) { BaseType::insert(BaseType::begin() + i, value); }

    // iterator insert(iterator before, int count, const T &value)

    //void insert(int i, int count, const T &value)
    //{
    //}

    // iterator insert(iterator before, const T &value)

    bool isEmpty() const { return BaseType::empty(); }
    T &last() { return BaseType::back(); }
    const T &last() const { return BaseType::back(); }

    //int lastIndexOf(const T &value, int from = -1) const
    //{}

    size_t length() const { return BaseType::size(); }

    //Vector<T> mid(int pos, int length = -1) const
    //{}

    //void move(int from, int to)
    //{}

    // void pop_back()
    // void pop_front()

    void prepend(const T &value) { BaseType::insert(BaseType::begin(), value); }

    // void push_back(const T &value)

    void push_front(const T &value) { prepend(value); }

    //void remove(int i)
    //{}

    //void remove(int i, int count)
    //{}

    //int removeAll(const T &value)
    //{}

    void removeAt(int i) { BaseType::erase(BaseType::begin() + i); }
    void removeFirst() { BaseType::erase(BaseType::begin()); }
    void removeLast() { BaseType::pop_back(); }

    //bool removeOne(const T &value)
    //{}

    //void replace(int i, const T &value)
    //{}

    // void reserve(int size)
    // void resize(int size)

    //int size() const { return (int)BaseType::size(); }
    //size_t size() const;

    void squeeze() { BaseType(*this).swap(*this); }    ///

    bool startsWith(const T &value) const { return (!BaseType::empty() && BaseType::front() == value); }

    //void swap(int i, int j)

    //T takeAt(int i)

    //T takeFirst()

    //T takeLast()

    T value(int i) const
    {
        if (i < 0 || i >= (int)BaseType::size())
            return T();
        return BaseType::at((size_t)i);
    }

    T value(int i, const T &defaultValue) const
    {
        if (i < 0 || i >= (int)BaseType::size())
            return defaultValue;
        return BaseType::at((size_t)i);
    }

    bool operator != (const Vector<T> &other) const { return !(*this == other); }

    //Vector<T> operator + (const Vector<T> &other) const
    //+=
    //+=
    //<<
    //<<

    Vector<T> &operator = (const Vector<T> &other) { BaseType::operator = (other); return *this; }
    bool operator == (const Vector<T> &other) const
    {
        if (BaseType::size() != other.BaseType::size())
            return false;
        for (size_t i = 0; i < BaseType::size(); i++)
            if (!(BaseType::at(i) == other.BaseType::at(i)))
                return false;
        return true;
    }

    // T &operator [] (int i)
    // const T &operator [] (int i) const

    ////
    //iterator
};

template <typename T>
class Deque : public std::deque<T>
{
};

#ifndef NO_CXX11_STL

/*template <typename T>
struct Hasher : private std::hash<unsigned int>
{
    // generic hasher (may not work for some classes)
    size_t operator () (const T &key) const
    {
        //printf("*");
        return std::hash<unsigned int>::operator () (*(reinterpret_cast<const unsigned int *>(&key)));
    }
};

template <typename T>
struct Hasher<T *> : private std::hash<void *>
{
    size_t operator () (T *key) const
    {
        //printf("&");
        return std::hash<void *>::operator () (reinterpret_cast<void *>(key));
    }
};

template <typename T>
struct Hasher<const T *> : private std::hash<const void *>
{
    size_t operator () (const T *key) const
    {
        //printf("$");
        return std::hash<const void *>::operator () (reinterpret_cast<const void *>(key));
    }
};

template <> struct Hasher<char> : public std::hash<char> {};
template <> struct Hasher<unsigned char> : public std::hash<unsigned char> {};
template <> struct Hasher<signed char> : public std::hash<signed char> {};
template <> struct Hasher<short> : public std::hash<short> {};
template <> struct Hasher<unsigned short> : public std::hash<unsigned short> {};
template <> struct Hasher<int> : public std::hash<int> {};
template <> struct Hasher<unsigned int> : public std::hash<unsigned int> {};
template <> struct Hasher<long> : public std::hash<long> {};
template <> struct Hasher<unsigned long> : public std::hash<unsigned long> {};
template <> struct Hasher<bool> : public std::hash<bool> {};
template <> struct Hasher<float> : public std::hash<float> {};
template <> struct Hasher<double> : public std::hash<double> {};
template <> struct Hasher<long double> : public std::hash<long double> {};

template <> struct Hasher<std::string> : public std::hash<std::string> {};*/

template <typename Key, typename T>
//class Hash : public std::unordered_map<Key, T, Hasher<Key> >
class Hash : public std::unordered_map<Key, T>
{
private:
    //typedef std::unordered_map<Key, T, Hasher<Key> > BaseType;
    typedef std::unordered_map<Key, T> BaseType;

public:
    // Hash()
    // Hash(const Hash<Key, T> &other)
    // ~Hash()
    // iterator begin()
    // const_iterator begin() const

    int capacity() const { return BaseType::bucket_count(); }

    // void clear()

    typename BaseType::const_iterator constBegin() const { return BaseType::cbegin(); }
    typename BaseType::const_iterator constEnd() const { return BaseType::cend(); }
    typename BaseType::const_iterator constFind(const Key &key) const { return BaseType::find(key); }
    bool contains(const Key &key) const { return (BaseType::find(key) != BaseType::end()); }

    // int count(const Key &key) const

    //int count() const { return size(); }
    size_t count() const { return BaseType::size(); }

    // bool empty() const
    // iterator end()
    // const_iterator end() const
    // iterator erase(iterator pos)
    // iterator find(const Key &key)
    // const_iterator find(const Key &key) const

    //iterator insert(const Key &key, const T &value) {
    //iterator insertMulti(const Key &key, const T &value) {

    bool isEmpty() const { return BaseType::empty(); }

    //const Key key(const T &value) const {
    //const Key key(const T &value, const Key &defaultKey) const {
    //List<Key> keys() const {
    //List<Key> keys(const T &value) const {
    //int remove(const Key &key) {

    // void reserve(int size)

    //int size() const { return (int)BaseType::size(); }
    //size_t size() const;

    //void squeeze() {

    //// static_cast?
    void swap(Hash<Key, T> &other) { BaseType::swap(*(static_cast<BaseType *>(&other))); }

    //T take(const Key &key) {
    //List<Key> uniqueKeys() const {
    //Hash<Key, T> &unite(const Hash<Key, T> &other) {

    //const T value(const Key &key) const
    //const T value(const Key &key, const T &defaultValue) const
    //List<T> values() const
    //List<T> values(const Key &key) const

    //bool operator != (const Hash<Key, T> &other) const

    Hash<Key, T> &operator = (const Hash<Key, T> &other) { BaseType::operator = (other); return *this; }

    //bool operator == (const Hash<Key, T> &other) const

    // T &operator [] (const Key &key)
    // const T &operator [] (const Key &key) const

    class iterator : public BaseType::iterator
    {
    public:
        iterator() : BaseType::iterator() {}
        iterator(typename BaseType::iterator it) : BaseType::iterator(it) {}
        const Key &key() const { return BaseType::iterator::operator * ().first; }
        T &value() const { return BaseType::iterator::operator * ().second; }

        ////
        bool operator != (const iterator &other) const { return BaseType::iterator::operator != (other); }
        //bool operator != (const)
        T &operator * () const { return value(); }
        iterator  operator +  (int j) const { return iterator(BaseType::iterator::operator + (j)); }
        iterator &operator ++ () { BaseType::iterator::operator ++ (); return *this; }
        iterator  operator ++ (int) { return iterator(BaseType::iterator::operator ++ (0)); }
        iterator &operator += (int j) { BaseType::iterator::operator += (j); return *this; }
        iterator  operator -  (int j) const { return iterator(BaseType::iterator::operator - (j)); }
        iterator &operator -- () { BaseType::iterator::operator -- (); return *this; }
        iterator  operator -- (int) { return iterator(BaseType::iterator::operator -- (0)); }
        iterator &operator -= (int j) { BaseType::iterator::operator -= (j); return *this; }
        T *operator -> () const { return &value(); }
        iterator &operator = (const iterator &other) { BaseType::iterator::operator = (other); return *this; }
        bool operator == (const iterator &other) const { return BaseType::iterator::operator == (other); }
        //bool operator == (const)
    private:
        //std::unordered_map<Key, T, Hasher<Key> >::iterator _it;

    };

    /*class const_iterator : public std::unordered_map<Key, T, Hasher<Key> >::const_iterator
    {
    public:
        const_iterator() : std::unordered_map<Key, T, Hasher<Key> >::const_iterator() {}

    };*/
};

/*template <typename Key, typename T>
class HashIterator : public std::unordered_map<Key, T, hasher<Key> >::iterator
{
public:
    //iterator() : _it() {}
    //iterator(std::unordered_map<Key, T, Hasher<Key> >::iterator it) : _it(it) {}
private:
    //std::unordered_map<Key, T, Hasher<Key> >::iterator _it;

};*/

/*template <typename Key, typename T>
class HashIterator
{
public:
    HashIterator() : _it() {}
    HashIterator(typename Hash<Key, T>::iterator it) : _it(it) {}
private:
    typename Hash<Key, T>::iterator _it;
};*/

template <typename Key, typename T>
class HashIterator : public Hash<Key, T>::iterator
{
public:
    HashIterator() : Hash<Key, T>::iterator() {}
    HashIterator(typename Hash<Key, T>::iterator it) : Hash<Key, T>::iterator(it) {}
private:
    //typename Hash<Key, T>::iterator _it;
};

#else // NO_CXX11_STL

template <typename Key, typename T>
class Hash : public std::map<Key, T>    // use map if no C++11 STL
{
private:
    typedef std::map<Key, T> BaseType;

public:
    // Hash()
    // Hash(const Hash<Key, T> &other)
    // ~Hash()
    // iterator begin()
    // const_iterator begin() const

    //int capacity() const { return BaseType::bucket_count(); }

    // void clear()

    typename BaseType::const_iterator constBegin() const { return BaseType::begin(); }
    typename BaseType::const_iterator constEnd() const { return BaseType::end(); }
    typename BaseType::const_iterator constFind(const Key &key) const { return BaseType::find(key); }
    bool contains(const Key &key) const { return (BaseType::find(key) != BaseType::end()); }

    // int count(const Key &key) const

    //int count() const { return size(); }
    size_t count() const { return BaseType::size(); }

    // bool empty() const
    // iterator end()
    // const_iterator end() const
    // iterator erase(iterator pos)
    // iterator find(const Key &key)
    // const_iterator find(const Key &key) const

    //iterator insert(const Key &key, const T &value) {
    //iterator insertMulti(const Key &key, const T &value) {

    bool isEmpty() const { return BaseType::empty(); }

    //const Key key(const T &value) const {
    //const Key key(const T &value, const Key &defaultKey) const {
    //List<Key> keys() const {
    //List<Key> keys(const T &value) const {
    //int remove(const Key &key) {

    // void reserve(int size)

    //int size() const { return (int)BaseType::size(); }
    //size_t size() const;

    //void squeeze() {

    //// static_cast?
    void swap(Hash<Key, T> &other) { BaseType::swap(*(static_cast<BaseType *>(&other))); }

    //T take(const Key &key) {
    //List<Key> uniqueKeys() const {
    //Hash<Key, T> &unite(const Hash<Key, T> &other) {

    //const T value(const Key &key) const
    //const T value(const Key &key, const T &defaultValue) const
    //List<T> values() const
    //List<T> values(const Key &key) const

    //bool operator != (const Hash<Key, T> &other) const

    Hash<Key, T> &operator = (const Hash<Key, T> &other) { BaseType::operator = (other); return *this; }

    //bool operator == (const Hash<Key, T> &other) const

    // T &operator [] (const Key &key)
    // const T &operator [] (const Key &key) const

    class iterator : public BaseType::iterator
    {
    public:
        iterator() : BaseType::iterator() {}
        iterator(typename BaseType::iterator it) : BaseType::iterator(it) {}
        const Key &key() const { return BaseType::iterator::operator * ().first; }
        T &value() const { return BaseType::iterator::operator * ().second; }

        ////
        bool operator != (const iterator &other) const { return BaseType::iterator::operator != (other); }
        //bool operator != (const)
        T &operator * () const { return value(); }
        iterator  operator +  (int j) const { return iterator(BaseType::iterator::operator + (j)); }
        iterator &operator ++ () { BaseType::iterator::operator ++ (); return *this; }
        iterator  operator ++ (int) { return iterator(BaseType::iterator::operator ++ (0)); }
        iterator &operator += (int j) { BaseType::iterator::operator += (j); return *this; }
        iterator  operator -  (int j) const { return iterator(BaseType::iterator::operator - (j)); }
        iterator &operator -- () { BaseType::iterator::operator -- (); return *this; }
        iterator  operator -- (int) { return iterator(BaseType::iterator::operator -- (0)); }
        iterator &operator -= (int j) { BaseType::iterator::operator -= (j); return *this; }
        T *operator -> () const { return &value(); }
        iterator &operator = (const iterator &other) { BaseType::iterator::operator = (other); return *this; }
        bool operator == (const iterator &other) const { return BaseType::iterator::operator == (other); }
        //bool operator == (const)
    private:
        //std::unordered_map<Key, T, Hasher<Key> >::iterator _it;

    };

    /*class const_iterator : public std::unordered_map<Key, T, Hasher<Key> >::const_iterator
    {
    public:
        const_iterator() : std::unordered_map<Key, T, Hasher<Key> >::const_iterator() {}

    };*/
};

#endif // NO_CXX11_STL

#endif // CONTAINERS_H
