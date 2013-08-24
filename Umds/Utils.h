#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

namespace utils {
    static vector<string>& tokenize(const string &s, char delim, vector<string> &tokens) {
        stringstream ss(s);
        string token;
        while (getline(ss, token, delim)) {
            tokens.push_back(token);
        }
        return tokens;
    }

    static vector<string> tokenize(const string &s, char delim) {
        vector<string> tokens;
        tokenize(s, delim, tokens);
        return tokens;
    }
}

#endif // UTILS_H
