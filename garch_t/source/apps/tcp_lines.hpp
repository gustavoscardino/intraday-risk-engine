// Line-oriented TCP client for the live feed (data/tools/feed_server.py): connects to HOST:PORT and
// returns the received text one line at a time. Winsock on Windows, BSD sockets elsewhere.
#pragma once

#include <stdexcept>
#include <string>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

class TcpLines {
public:
    // address: "HOST:PORT" (e.g. "127.0.0.1:9000"). Throws std::runtime_error if it cannot connect.
    explicit TcpLines(const std::string& address) {
        const size_t colon = address.rfind(':');
        if (colon == std::string::npos || colon == 0 || colon + 1 == address.size())
            throw std::runtime_error("--live expects HOST:PORT, got '" + address + "'");
        const std::string host = address.substr(0, colon), port = address.substr(colon + 1);
#if defined(_WIN32)
        WSADATA wsa;
        if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) throw std::runtime_error("cannot start Winsock");
        started_ = true;
#endif
        addrinfo hints{};
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        addrinfo* found = nullptr;
        if (getaddrinfo(host.c_str(), port.c_str(), &hints, &found) != 0 || !found) {
            close_all();
            throw std::runtime_error("cannot resolve " + address);
        }
        for (addrinfo* a = found; a; a = a->ai_next) {
            sock_ = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
            if (!valid()) continue;
            if (connect(sock_, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0) break;
            close_socket();
        }
        freeaddrinfo(found);
        if (!valid()) {
            close_all();
            throw std::runtime_error("cannot connect to " + address + " (is the feed server running?)");
        }
    }
    ~TcpLines() { close_all(); }
    TcpLines(const TcpLines&) = delete;
    TcpLines& operator=(const TcpLines&) = delete;

    // Next line without its end-of-line; false when the server closed the connection.
    bool next(std::string& line) {
        for (;;) {
            const size_t eol = buf_.find('\n', pos_);
            if (eol != std::string::npos) {
                line.assign(buf_, pos_, eol - pos_);
                pos_ = eol + 1;
                if (pos_ > (1u << 16)) {
                    buf_.erase(0, pos_);
                    pos_ = 0;
                }
                return true;
            }
            char chunk[1 << 14];
            const auto got = recv(sock_, chunk, static_cast<int>(sizeof chunk), 0);
            if (got <= 0) {
                if (pos_ < buf_.size()) {  // a last line without its end-of-line
                    line.assign(buf_, pos_, std::string::npos);
                    pos_ = buf_.size();
                    return true;
                }
                return false;
            }
            buf_.append(chunk, static_cast<size_t>(got));
        }
    }

private:
#if defined(_WIN32)
    SOCKET sock_ = INVALID_SOCKET;
    bool started_ = false;
    bool valid() const { return sock_ != INVALID_SOCKET; }
    void close_socket() {
        if (valid()) closesocket(sock_);
        sock_ = INVALID_SOCKET;
    }
    void close_all() {
        close_socket();
        if (started_) WSACleanup();
        started_ = false;
    }
#else
    int sock_ = -1;
    bool valid() const { return sock_ >= 0; }
    void close_socket() {
        if (valid()) ::close(sock_);
        sock_ = -1;
    }
    void close_all() { close_socket(); }
#endif
    std::string buf_;
    size_t pos_ = 0;
};
