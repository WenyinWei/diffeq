# DiffEq Documentation System

This directory contains the comprehensive documentation system for the DiffEq C++ library. The documentation is automatically generated and includes multiple formats and tools for different use cases.

## 📚 Documentation Structure

```
docs/
├── README.md                    # This file
├── index.md                     # Main documentation index
├── api/                         # API documentation
│   ├── README.md               # API overview
│   ├── integrators/            # Integrator documentation
│   ├── core/                   # Core concepts
│   └── interfaces/             # Interface documentation
├── examples/                    # Examples documentation
│   └── README.md               # Examples overview
├── performance/                 # Performance guides
│   └── README.md               # Performance overview
├── templates/                   # Documentation templates
│   └── header_template.md      # Header documentation template
├── generated/                   # Auto-generated documentation
│   ├── html/                   # Doxygen HTML output
│   ├── xml/                    # Doxygen XML output
│   └── latex/                  # Doxygen LaTeX output
└── sphinx/                     # Sphinx documentation
    ├── conf.py                 # Sphinx configuration
    ├── index.rst               # Sphinx index
    └── _build/                 # Sphinx build output
```

## 🚀 Quick Start

### Generate All Documentation

```bash
# Using the build script
./tools/scripts/build_docs.sh all

# Using xmake
xmake docs

# Using the Windows batch script
tools\scripts\build_docs.bat all
```

### Generate Specific Documentation

```bash
# Doxygen only
./tools/scripts/build_docs.sh doxygen

# Sphinx only
./tools/scripts/build_docs.sh sphinx

# API documentation only
./tools/scripts/build_docs.sh api

# Examples documentation only
./tools/scripts/build_docs.sh examples

# Performance documentation only
./tools/scripts/build_docs.sh performance
```

### Serve Documentation Locally

```bash
# Generate and serve documentation
./tools/scripts/build_docs.sh serve

# This will start a local HTTP server at http://localhost:8000
```

## 🛠️ Documentation Tools

### Doxygen

**Purpose**: Generate comprehensive API documentation from source code comments.

**Features**:
- Automatic extraction of classes, functions, and templates
- Cross-references and inheritance diagrams
- Search functionality
- Multiple output formats (HTML, XML, LaTeX)

**Configuration**: `Doxyfile` in the project root

**Output**: `docs/generated/html/index.html`

### Sphinx

**Purpose**: Generate user guides, tutorials, and high-level documentation.

**Features**:
- Rich text formatting with reStructuredText
- Integration with Doxygen via Breathe
- Multiple themes and customization
- Search and navigation

**Configuration**: `docs/sphinx/conf.py`

**Output**: `docs/sphinx/_build/html/index.html`

### Custom Documentation Generator

**Purpose**: Generate structured documentation from templates.

**Features**:
- Template-based generation
- Automatic header file parsing
- Consistent documentation structure
- Markdown output

**Script**: `tools/scripts/generate_header_docs.py`

## 📝 Writing Documentation

### Header File Documentation

Use the standard Doxygen format for documenting C++ code:

```cpp
/**
 * @brief Brief description of the class/function
 * 
 * Detailed description of the class/function and its purpose.
 * 
 * @tparam T Template parameter description
 * @param x Parameter description
 * @return Return value description
 * 
 * @example
 * ```cpp
 * // Example usage
 * MyClass<int> obj;
 * obj.method(42);
 * ```
 */
template<typename T>
class MyClass {
public:
    /**
     * @brief Method description
     * @param x Parameter description
     * @return Return description
     */
    T method(T x);
};
```

### Markdown Documentation

For user guides and tutorials, use Markdown with the following conventions:

- Use `#` for main titles
- Use `##` for section headers
- Use `###` for subsection headers
- Use `**bold**` for emphasis
- Use `*italic*` for secondary emphasis
- Use ```cpp for code blocks
- Use `-` for unordered lists
- Use `1.` for ordered lists

### Documentation Templates

Use the provided templates for consistency:

- `docs/templates/header_template.md` - For header file documentation
- Custom templates can be created for specific documentation types

## 🔧 Configuration

### Doxygen Configuration

The main Doxygen configuration is in `Doxyfile`. Key settings:

- `PROJECT_NAME`: "DiffEq - Modern C++ ODE Integration Library"
- `INPUT`: Source directories to process
- `OUTPUT_DIRECTORY`: `docs/generated`
- `GENERATE_HTML`: `YES`
- `GENERATE_LATEX`: `NO`
- `HAVE_DOT`: `YES` (for diagrams)

### Sphinx Configuration

Sphinx configuration is in `docs/sphinx/conf.py`. Key settings:

- `project`: 'DiffEq'
- `extensions`: Includes Breathe and Exhale for Doxygen integration
- `html_theme`: 'sphinx_rtd_theme'
- `breathe_projects`: Maps to Doxygen XML output

### Build Scripts

- `tools/scripts/build_docs.sh` - Unix/Linux/macOS build script
- `tools/scripts/build_docs.bat` - Windows build script
- Both scripts support the same command-line options

## 🚀 Continuous Integration

The documentation is automatically built and deployed via GitHub Actions:

- **Trigger**: Push to `main` or `develop` branches
- **Build**: Generates all documentation formats
- **Deploy**: Publishes to GitHub Pages
- **Check**: Validates documentation quality

### GitHub Actions Workflow

The workflow (`/.github/workflows/docs.yml`) includes:

1. **Build Documentation**: Generates Doxygen, Sphinx, and custom docs
2. **Deploy to GitHub Pages**: Publishes documentation automatically
3. **Quality Check**: Validates documentation structure and syntax

## 📊 Documentation Quality

### Automated Checks

The documentation system includes automated quality checks:

- **Syntax Validation**: Ensures proper Doxygen/Sphinx syntax
- **Structure Validation**: Verifies required documentation files exist
- **Link Validation**: Checks for broken internal links
- **Coverage Analysis**: Identifies undocumented components

### Manual Review Checklist

Before publishing documentation:

- [ ] All public APIs are documented
- [ ] Examples compile and run correctly
- [ ] Links and cross-references work
- [ ] Code examples follow library conventions
- [ ] Performance considerations are noted
- [ ] Thread safety is documented
- [ ] Error handling is explained

## 🔍 Troubleshooting

### Common Issues

**Doxygen fails to generate diagrams**
- Install Graphviz: `sudo apt-get install graphviz`
- Ensure `HAVE_DOT = YES` in Doxyfile

**Sphinx build fails**
- Install Python dependencies: `pip install sphinx sphinx-rtd-theme breathe exhale`
- Check `docs/sphinx/conf.py` configuration

**Template generation fails**
- Ensure Python 3.6+ is installed
- Check template file paths and permissions

**GitHub Pages not updating**
- Verify GitHub Actions workflow is running
- Check repository settings for GitHub Pages
- Ensure `gh-pages` branch exists

### Getting Help

1. Check the build logs for specific error messages
2. Verify all dependencies are installed
3. Ensure file paths and permissions are correct
4. Consult the tool-specific documentation:
   - [Doxygen Manual](https://www.doxygen.nl/manual/)
   - [Sphinx Documentation](https://www.sphinx-doc.org/)
   - [Breathe Documentation](https://breathe.readthedocs.io/)

## 🤝 Contributing to Documentation

### Adding New Documentation

1. **Header Files**: Add Doxygen comments to new header files
2. **User Guides**: Create Markdown files in appropriate directories
3. **Examples**: Add example code with documentation
4. **Templates**: Extend templates for new documentation types

### Documentation Standards

- Use clear, concise language
- Include practical examples
- Document all public APIs
- Follow the established style guide
- Test all code examples
- Update documentation with code changes

### Review Process

1. Create documentation changes in a feature branch
2. Ensure all automated checks pass
3. Review generated documentation locally
4. Submit pull request with documentation changes
5. Address review feedback
6. Merge when approved

## 📈 Performance and Optimization

### Build Performance

- **Incremental Builds**: Only regenerate changed documentation
- **Parallel Processing**: Use multiple cores for large documentation sets
- **Caching**: Cache intermediate results when possible

### Output Optimization

- **Minification**: Compress HTML/CSS/JS output
- **Image Optimization**: Compress diagrams and images
- **CDN Integration**: Use CDN for static assets

## 🔮 Future Enhancements

### Planned Features

- **Interactive Examples**: Live code examples with online compilation
- **API Versioning**: Support for multiple API versions
- **Search Enhancement**: Advanced search with filters
- **Mobile Optimization**: Better mobile documentation experience
- **Dark Mode**: Dark theme for documentation
- **PDF Generation**: Automated PDF documentation generation

### Integration Opportunities

- **IDE Integration**: Documentation in IDEs and editors
- **API Documentation**: Integration with API documentation services
- **Community Features**: User comments and feedback system
- **Analytics**: Documentation usage analytics

---

For more information about the DiffEq library, see the [main README](../README.md) and [API documentation](api/README.md). 