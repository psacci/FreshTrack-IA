const http = require('http');
const fs = require('fs');
const path = require('path');
const mime = require('mime-types');

const PORT = process.env.PORT || 3000;

const publicDirectory = path.join(__dirname, 'public');
const uploadDirectory = path.join(__dirname, 'uploads');

const allowedExtensions = [
    '.txt',
    '.pdf',
    '.jpg',
    '.jpeg',
    '.png'
];

const server = http.createServer((req, res) => {

    // Handle file uploads
    if (req.url === '/upload' && req.method === 'POST') {
        handleUpload(req, res);
        return;
    }

    // Serve files from the public directory
    let requestedFile = req.url === '/'
        ? 'index.html'
        : decodeURIComponent(req.url);

    const filePath = path.normalize(
        path.join(publicDirectory, requestedFile)
    );

    // Prevent access outside the public folder
    if (!filePath.startsWith(publicDirectory)) {
        res.writeHead(403, { 'Content-Type': 'text/html' });
        res.end('<h1>403 - Forbidden</h1>');
        return;
    }

    fs.readFile(filePath, (err, content) => {

        if (err) {

            if (err.code === 'ENOENT') {
                res.writeHead(404, {
                    'Content-Type': 'text/html'
                });

                res.end(`
                    <!DOCTYPE html>
                    <html>
                    <head>
                        <title>404 - Not Found</title>
                    </head>
                    <body>
                        <h1>404 - File Not Found</h1>
                        <p>The requested file could not be found.</p>
                    </body>
                    </html>
                `);

            } else {
                console.error(err);

                res.writeHead(500, {
                    'Content-Type': 'text/html'
                });

                res.end('<h1>500 - Internal Server Error</h1>');
            }

            return;
        }

        const contentType =
            mime.lookup(filePath) || 'application/octet-stream';

        res.writeHead(200, {
            'Content-Type': contentType
        });

        res.end(content);
    });
});

function handleUpload(req, res) {

    let body = Buffer.alloc(0);

    req.on('data', chunk => {
        body = Buffer.concat([body, chunk]);
    });

    req.on('end', () => {

        const contentType = req.headers['content-type'];

        if (!contentType || !contentType.includes('multipart/form-data')) {
            res.writeHead(400, {
                'Content-Type': 'text/html'
            });

            res.end('<h1>400 - Invalid Upload Request</h1>');
            return;
        }

        const boundaryMatch = contentType.match(/boundary=(?:"([^"]+)"|([^;]+))/);

        if (!boundaryMatch) {
            res.writeHead(400, {
                'Content-Type': 'text/html'
            });

            res.end('<h1>400 - Invalid Form Data</h1>');
            return;
        }

        const boundary = boundaryMatch[1] || boundaryMatch[2];
        const boundaryBuffer = Buffer.from(`--${boundary}`);

        const parts = body
            .toString('binary')
            .split(boundaryBuffer.toString('binary'));

        let uploadedFile = null;

        for (const part of parts) {

            if (!part.includes('filename=')) {
                continue;
            }

            const filenameMatch =
                part.match(/filename="([^"]*)"/);

            if (!filenameMatch || !filenameMatch[1]) {
                continue;
            }

            const originalName = path.basename(filenameMatch[1]);
            const extension = path.extname(originalName).toLowerCase();

            if (!allowedExtensions.includes(extension)) {

                res.writeHead(400, {
                    'Content-Type': 'text/html'
                });

                res.end(`
                    <h1>Upload Failed</h1>
                    <p>File type <strong>${extension || 'unknown'}</strong> is not allowed.</p>
                    <p>Allowed types: TXT, PDF, JPG, JPEG, PNG.</p>
                `);

                return;
            }

            const headerEnd = part.indexOf('\r\n\r\n');

            if (headerEnd === -1) {
                continue;
            }

            let fileData = part.substring(headerEnd + 4);

            fileData = fileData.replace(/\r\n$/, '');

            uploadedFile = {
                name: originalName,
                extension: extension,
                data: Buffer.from(fileData, 'binary')
            };

            break;
        }

        if (!uploadedFile) {

            res.writeHead(400, {
                'Content-Type': 'text/html'
            });

            res.end(`
                <h1>Upload Failed</h1>
                <p>No file was selected.</p>
            `);

            return;
        }

        const safeName =
            `${Date.now()}-${uploadedFile.name}`;

        const destination =
            path.join(uploadDirectory, safeName);

        fs.writeFile(destination, uploadedFile.data, err => {

            if (err) {
                console.error(err);

                res.writeHead(500, {
                    'Content-Type': 'text/html'
                });

                res.end('<h1>500 - Could Not Save File</h1>');
                return;
            }

            res.writeHead(200, {
                'Content-Type': 'text/html'
            });

            res.end(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Upload Successful</title>
                </head>
                <body>
                    <h1>Upload Successful</h1>
                    <p><strong>${uploadedFile.name}</strong> was uploaded successfully.</p>
                    <a href="/">Return to File Server</a>
                </body>
                </html>
            `);
        });
    });
}

server.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});